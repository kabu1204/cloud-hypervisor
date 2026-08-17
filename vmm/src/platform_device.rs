// SPDX-License-Identifier: Apache-2.0
//
// VFIO platform device passthrough (e.g. noiommu groups on ARM platforms).
//
// Unlike the PCI VFIO path, platform devices are opened directly through
// their VFIO group (no container / no type1 IOMMU), their MMIO regions are
// mapped into the guest at fixed guest physical addresses and their (level)
// interrupts are wired to the in-kernel IRQ chip through irqfd/resamplefd.

use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io;
use std::mem::size_of;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hypervisor::Vm;
use log::{info, warn};
use vfio_bindings::bindings::vfio::*;
use vm_allocator::MemorySlotAllocator;
use vm_allocator::page_size::get_page_size;
use vm_memory::MmapRegion;
use vmm_sys_util::eventfd::EventFd;
use vmm_sys_util::ioctl::{ioctl_with_mut_ref, ioctl_with_ptr, ioctl_with_ref, ioctl_with_val};
use vmm_sys_util::ioctl_io_nr;

use crate::vm_config::PlatformDeviceConfig;

// The VFIO ioctl request numbers (mirroring <linux/vfio.h>). The kernel
// dispatches on the nr only; they are declared like vfio-ioctls does.
ioctl_io_nr!(VFIO_SET_IOMMU, VFIO_TYPE.into(), VFIO_BASE + 2);
ioctl_io_nr!(VFIO_GROUP_GET_STATUS, VFIO_TYPE.into(), VFIO_BASE + 3);
ioctl_io_nr!(VFIO_GROUP_SET_CONTAINER, VFIO_TYPE.into(), VFIO_BASE + 4);
ioctl_io_nr!(VFIO_GROUP_GET_DEVICE_FD, VFIO_TYPE.into(), VFIO_BASE + 6);
ioctl_io_nr!(VFIO_DEVICE_GET_INFO, VFIO_TYPE.into(), VFIO_BASE + 7);
ioctl_io_nr!(VFIO_DEVICE_GET_REGION_INFO, VFIO_TYPE.into(), VFIO_BASE + 8);
ioctl_io_nr!(VFIO_DEVICE_GET_IRQ_INFO, VFIO_TYPE.into(), VFIO_BASE + 9);
ioctl_io_nr!(VFIO_DEVICE_SET_IRQS, VFIO_TYPE.into(), VFIO_BASE + 10);

/// Platform device IRQ index (vfio-platform exposes its IRQs starting at 0).
const VFIO_PLATFORM_IRQ_INDEX: u32 = 0;

pub type Result<T> = std::result::Result<T, PlatformDeviceError>;

#[derive(Debug)]
pub enum PlatformDeviceError {
    OpenGroup(PathBuf, io::Error),
    OpenContainer(io::Error),
    GroupStatus(io::Error),
    GroupNotViable,
    SetContainer(io::Error),
    SetNoiommu(io::Error),
    ResolveDeviceName(PathBuf, String),
    GetDeviceFd(io::Error),
    GetDeviceInfo(io::Error),
    GetRegionInfo(io::Error),
    MmapRegion(io::Error),
    CreateUserMemoryRegion(hypervisor::HypervisorVmError),
    GetIrqInfo(io::Error),
    SetIrqs(io::Error),
    CreateEventFd(io::Error),
    RegisterIrqfd(hypervisor::HypervisorVmError),
}

impl std::fmt::Display for PlatformDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PlatformDeviceError {}

/// Open a VFIO (noiommu) group and attach it to a fresh noiommu container.
///
/// On recent kernels (>= 6.16) a noiommu group still requires a container
/// with the NOIOMMU iommu type set before devices can be opened from it.
/// Returns (container, group); both must stay open for the device lifetime.
pub fn open_group(group_path: &Path) -> Result<(File, File)> {
    let container = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/vfio/vfio")
        .map_err(PlatformDeviceError::OpenContainer)?;

    let group = OpenOptions::new()
        .read(true)
        .write(true)
        .open(group_path)
        .map_err(|e| PlatformDeviceError::OpenGroup(group_path.to_path_buf(), e))?;

    let mut status = vfio_group_status {
        argsz: size_of::<vfio_group_status>() as u32,
        flags: 0,
    };
    // SAFETY: `group` is a valid VFIO group fd and `status` is a valid
    // vfio_group_status with argsz set.
    let ret = unsafe { ioctl_with_mut_ref(&group, VFIO_GROUP_GET_STATUS(), &mut status) };
    if ret < 0 {
        return Err(PlatformDeviceError::GroupStatus(io::Error::last_os_error()));
    }
    if status.flags & VFIO_GROUP_FLAGS_VIABLE == 0 {
        return Err(PlatformDeviceError::GroupNotViable);
    }

    // SAFETY: both fds are valid; the group is a noiommu group without a
    // container yet.
    let ret =
        unsafe { ioctl_with_ref(&group, VFIO_GROUP_SET_CONTAINER(), &(container.as_raw_fd())) };
    if ret < 0 {
        return Err(PlatformDeviceError::SetContainer(io::Error::last_os_error()));
    }
    // SAFETY: `container` is a valid VFIO container fd.
    let ret = unsafe { ioctl_with_val(&container, VFIO_SET_IOMMU(), VFIO_NOIOMMU_IOMMU.into()) };
    if ret < 0 {
        return Err(PlatformDeviceError::SetNoiommu(io::Error::last_os_error()));
    }

    Ok((container, group))
}

/// Resolve the platform device name (e.g. "fdad0000.npu") for a VFIO group
/// by looking at the matching /sys/kernel/iommu_groups/<id>/devices entry.
fn device_name_from_sysfs(group_path: &Path) -> Result<String> {
    let group_name = group_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let group_id = group_name
        .strip_prefix("noiommu-")
        .unwrap_or(group_name)
        .to_owned();
    let devices_dir = format!("/sys/kernel/iommu_groups/{group_id}/devices");
    let mut entries = std::fs::read_dir(&devices_dir)
        .map_err(|e| {
            PlatformDeviceError::ResolveDeviceName(devices_dir.clone().into(), e.to_string())
        })?
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned());
    match (entries.next(), entries.next()) {
        (Some(name), None) => Ok(name),
        _ => Err(PlatformDeviceError::ResolveDeviceName(
            devices_dir.into(),
            "expected exactly one device in group".to_owned(),
        )),
    }
}

/// Get a VFIO device fd from a group for the named device.
pub fn get_device_fd(group: &File, device_name: &str) -> Result<File> {
    let name = CString::new(device_name).unwrap();
    // SAFETY: `group` is a valid VFIO group fd and `name` is a valid
    // NUL-terminated string.
    let fd = unsafe {
        ioctl_with_ptr(
            group,
            VFIO_GROUP_GET_DEVICE_FD(),
            name.as_ptr() as *const u8,
        )
    };
    if fd < 0 {
        return Err(PlatformDeviceError::GetDeviceFd(io::Error::last_os_error()));
    }
    // SAFETY: the kernel returned a valid, owned VFIO device fd.
    Ok(unsafe { File::from_raw_fd(fd) })
}

pub struct VfioPlatformRegionInfo {
    pub index: u32,
    pub flags: u32,
    pub size: u64,
    pub offset: u64,
}

pub struct VfioPlatformIrqInfo {
    pub index: u32,
    pub flags: u32,
    pub count: u32,
}

/// Query device info (regions and IRQs) from a VFIO device fd.
pub fn get_device_info(
    device: &File,
) -> Result<(Vec<VfioPlatformRegionInfo>, Vec<VfioPlatformIrqInfo>)> {
    let mut dev_info = vfio_device_info {
        argsz: size_of::<vfio_device_info>() as u32,
        flags: 0,
        num_regions: 0,
        num_irqs: 0,
        cap_offset: 0,
        pad: 0,
    };
    // SAFETY: `device` is a valid VFIO device fd and `dev_info` is valid.
    let ret = unsafe { ioctl_with_mut_ref(device, VFIO_DEVICE_GET_INFO(), &mut dev_info) };
    if ret < 0 {
        return Err(PlatformDeviceError::GetDeviceInfo(
            io::Error::last_os_error(),
        ));
    }

    let mut regions = Vec::new();
    for index in 0..dev_info.num_regions {
        let mut region_info = vfio_region_info {
            argsz: size_of::<vfio_region_info>() as u32,
            flags: 0,
            index,
            cap_offset: 0,
            size: 0,
            offset: 0,
        };
        // SAFETY: `device` is a valid VFIO device fd and `region_info` is
        // valid with argsz set.
        let ret =
            unsafe { ioctl_with_mut_ref(device, VFIO_DEVICE_GET_REGION_INFO(), &mut region_info) };
        if ret < 0 {
            return Err(PlatformDeviceError::GetRegionInfo(
                io::Error::last_os_error(),
            ));
        }
        regions.push(VfioPlatformRegionInfo {
            index,
            flags: region_info.flags,
            size: region_info.size,
            offset: region_info.offset,
        });
    }

    let mut irqs = Vec::new();
    for index in 0..dev_info.num_irqs {
        let mut irq_info = vfio_irq_info {
            argsz: size_of::<vfio_irq_info>() as u32,
            flags: 0,
            index,
            count: 0,
        };
        // SAFETY: `device` is a valid VFIO device fd and `irq_info` is valid.
        let ret = unsafe { ioctl_with_mut_ref(device, VFIO_DEVICE_GET_IRQ_INFO(), &mut irq_info) };
        if ret < 0 {
            return Err(PlatformDeviceError::GetIrqInfo(io::Error::last_os_error()));
        }
        irqs.push(VfioPlatformIrqInfo {
            index,
            flags: irq_info.flags,
            count: irq_info.count,
        });
    }

    Ok((regions, irqs))
}

/// Set a VFIO IRQ action with an eventfd.
fn set_irq_eventfd(device: &File, irq_index: u32, action: u32, fd: RawFd) -> Result<()> {
    let irq_set = vfio_irq_set {
        argsz: (size_of::<vfio_irq_set>() + size_of::<u32>()) as u32,
        flags: VFIO_IRQ_SET_DATA_EVENTFD | action,
        index: irq_index,
        start: 0,
        count: 1,
        data: Default::default(),
    };
    let mut buf = Vec::with_capacity(irq_set.argsz as usize);
    // SAFETY: vfio_irq_set is a POD header; the u32 eventfd is appended right
    // after it as required by VFIO_IRQ_SET_DATA_EVENTFD.
    unsafe {
        buf.extend_from_slice(std::slice::from_raw_parts(
            &irq_set as *const vfio_irq_set as *const u8,
            size_of::<vfio_irq_set>(),
        ));
        buf.extend_from_slice(&(fd as u32).to_ne_bytes());
        // SAFETY: `device` is a valid VFIO device fd and `buf` is a valid
        // vfio_irq_set with argsz set.
        let ret = ioctl_with_ptr(device, VFIO_DEVICE_SET_IRQS(), buf.as_ptr());
        if ret < 0 {
            return Err(PlatformDeviceError::SetIrqs(io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// mmap a VFIO device region. The returned mapping owns the mapping and
/// unmaps it when dropped.
pub fn mmap_region(
    device: &File,
    region: &VfioPlatformRegionInfo,
    len: usize,
) -> Result<MmapRegion> {
    let mut prot = 0;
    if region.flags & VFIO_REGION_INFO_FLAG_READ != 0 {
        prot |= libc::PROT_READ;
    }
    if region.flags & VFIO_REGION_INFO_FLAG_WRITE != 0 {
        prot |= libc::PROT_WRITE;
    }
    // SAFETY: FFI call with a valid VFIO device fd; `region.offset` is the
    // kernel-provided mmap offset (region index encoded in the top bits).
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            prot,
            libc::MAP_SHARED,
            device.as_raw_fd(),
            region.offset as libc::off_t,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err(PlatformDeviceError::MmapRegion(io::Error::last_os_error()));
    }
    // SAFETY: `addr` points at the mapping of `len` bytes created just above,
    // with matching protection and flags.
    unsafe { MmapRegion::build_raw(addr.cast(), len, prot, libc::MAP_SHARED) }
        .map_err(|e| PlatformDeviceError::MmapRegion(io::Error::other(format!("{e:?}"))))
}

struct PlatformIrq {
    // Keep the eventfds open for the lifetime of the device; the kernel
    // (VFIO and KVM irqfd) holds references to them.
    #[allow(dead_code)]
    trigger: EventFd,
    #[allow(dead_code)]
    resample: EventFd,
    gsi: u32,
}

struct PlatformRegionMapping {
    slot: u32,
    gpa: u64,
    mapping: MmapRegion,
}

/// A VFIO platform device mapped into the guest address space.
pub struct VfioPlatformDevice {
    id: String,
    // Keep the fds open for the lifetime of the device.
    #[allow(dead_code)]
    container: File,
    #[allow(dead_code)]
    group: File,
    #[allow(dead_code)]
    device: File,
    regions: Vec<PlatformRegionMapping>,
    irq: Option<PlatformIrq>,
}

impl VfioPlatformDevice {
    pub fn new(
        config: &PlatformDeviceConfig,
        vm: &Arc<dyn Vm>,
        slot_allocator: &MemorySlotAllocator,
    ) -> Result<Self> {
        let (container, group) = open_group(&config.group)?;
        let device_name = match &config.device_name {
            Some(name) => name.clone(),
            None => device_name_from_sysfs(&config.group)?,
        };
        let device = get_device_fd(&group, &device_name)?;

        let (region_infos, irq_infos) = get_device_info(&device)?;
        info!(
            "platform device {} ({}): {} region(s), {} irq(s)",
            config.group.display(),
            device_name,
            region_infos.len(),
            irq_infos.len()
        );

        // Resolve the GPA of each region from the `map` parameter: either one
        // GPA per region, or a single base GPA with regions packed
        // contiguously (page-aligned).
        let page_size = get_page_size() as u64;
        let mut gpas = Vec::with_capacity(region_infos.len());
        if config.map.len() == region_infos.len() {
            gpas = config.map.clone();
        } else if config.map.len() == 1 && !region_infos.is_empty() {
            let mut next = config.map[0];
            for region in &region_infos {
                gpas.push(next);
                next += region.size.div_ceil(page_size) * page_size;
            }
        } else if !region_infos.is_empty() {
            warn!(
                "platform device {}: 'map' has {} entr(ies) but the device has {} region(s); \
                 no regions will be mapped",
                config.group.display(),
                config.map.len(),
                region_infos.len()
            );
        }

        let mut regions = Vec::new();
        for (region, gpa) in region_infos.iter().zip(gpas.iter()) {
            // KVM_SET_USER_MEMORY_REGION requires a page-size multiple.
            let len = (region.size.div_ceil(page_size) * page_size) as usize;
            let mapping = if region.flags & VFIO_REGION_INFO_FLAG_MMAP != 0 {
                mmap_region(&device, region, len)?
            } else {
                // The region is not mmap-able through VFIO (e.g. sub-page
                // register windows, which vfio-platform refuses to expose
                // for mmap). Fall back to mapping the physical address
                // directly through /dev/mem, assuming GPA == HPA (which
                // holds for the NPU passthrough use case).
                warn!(
                    "platform device {}: region {} (size {:#x}) is not mmap-able via VFIO; \
                     falling back to /dev/mem at PA=GPA {:#x}",
                    config.group.display(),
                    region.index,
                    region.size,
                    gpa
                );
                let devmem = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open("/dev/mem")
                    .map_err(PlatformDeviceError::MmapRegion)?;
                // SAFETY: FFI call with a valid fd; `gpa` is the region's
                // physical address in the identity-mapped setup.
                let addr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        devmem.as_raw_fd(),
                        *gpa as libc::off_t,
                    )
                };
                if addr == libc::MAP_FAILED {
                    warn!(
                        "platform device {}: region {} cannot be mapped at all; skipping",
                        config.group.display(),
                        region.index
                    );
                    continue;
                }
                // SAFETY: `addr` points at the mapping of `len` bytes
                // created just above.
                unsafe {
                    MmapRegion::build_raw(
                        addr.cast(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                    )
                }
                .map_err(|e| PlatformDeviceError::MmapRegion(io::Error::other(format!("{e:?}"))))?
            };
            let slot = slot_allocator.next_memory_slot();
            // SAFETY: MmapRegion invariants guarantee that `mapping.as_ptr()`
            // points at `len` valid bytes that stay mapped until the device
            // is dropped.
            unsafe {
                vm.create_user_memory_region(slot, *gpa, len, mapping.as_ptr(), false, false)
            }
            .map_err(PlatformDeviceError::CreateUserMemoryRegion)?;
            info!(
                "platform device {}: region {} mapped: GPA {:#x} -> host VA {:#x} (size {:#x})",
                config.group.display(),
                region.index,
                gpa,
                mapping.as_ptr() as u64,
                len
            );
            regions.push(PlatformRegionMapping {
                slot,
                gpa: *gpa,
                mapping,
            });
        }

        let irq = if let Some(spi) = config.irq {
            let irq_info = irq_infos
                .iter()
                .find(|i| i.index == VFIO_PLATFORM_IRQ_INDEX)
                .ok_or_else(|| {
                    PlatformDeviceError::GetIrqInfo(io::Error::other(format!(
                        "platform device has no IRQ index {VFIO_PLATFORM_IRQ_INDEX}"
                    )))
                })?;
            if irq_info.flags & VFIO_IRQ_INFO_EVENTFD == 0 {
                return Err(PlatformDeviceError::SetIrqs(io::Error::other(
                    "platform device IRQ does not support eventfd signaling",
                )));
            }
            let trigger =
                EventFd::new(libc::EFD_NONBLOCK).map_err(PlatformDeviceError::CreateEventFd)?;
            let resample =
                EventFd::new(libc::EFD_NONBLOCK).map_err(PlatformDeviceError::CreateEventFd)?;
            set_irq_eventfd(
                &device,
                irq_info.index,
                VFIO_IRQ_SET_ACTION_TRIGGER,
                trigger.as_raw_fd(),
            )?;
            set_irq_eventfd(
                &device,
                irq_info.index,
                VFIO_IRQ_SET_ACTION_UNMASK,
                resample.as_raw_fd(),
            )?;
            // The `irq` parameter is the *guest-visible virtual SPI*, not
            // the physical one — VFIO decouples them (physical IRQ -> vfio
            // trigger eventfd -> KVM irqfd -> vGIC SPI).
            //
            // Constraint: cloud-hypervisor only installs GSI routing for the
            // legacy SPI range (devices/src/gic.rs: gsi 32..63 -> SPI pins
            // 0..31, set up in Gic::enable() before platform devices are
            // created). KVM irqfd injection goes through that routing table,
            // so the virtual SPI MUST be in 0..31 or kvm_irq_map_gsi() finds
            // no entry and the interrupt is silently dropped (verified: jobs
            // completed on hardware, IRQ fired on the host, but the guest
            // never saw it).
            if spi >= 32 {
                return Err(PlatformDeviceError::SetIrqs(io::Error::other(
                    format!("platform device IRQ must be a legacy-range SPI (0..31), got {spi}"),
                )));
            }
            let gsi = spi + 32;
            eprintln!(
                "NPUDEBUG: register_irqfd_with_resample trigger_fd={} resample_fd={} gsi={}",
                trigger.as_raw_fd(),
                resample.as_raw_fd(),
                gsi
            );
            vm.register_irqfd_with_resample(&trigger, &resample, gsi)
                .map_err(PlatformDeviceError::RegisterIrqfd)?;
            info!(
                "platform device {}: virtual SPI {spi} (guest INTID {gsi}) wired via irqfd+resamplefd",
                config.group.display()
            );
            Some(PlatformIrq {
                trigger,
                resample,
                gsi,
            })
        } else {
            None
        };

        Ok(VfioPlatformDevice {
            id: config.id.clone().unwrap_or_else(|| device_name.clone()),
            container,
            group,
            device,
            regions,
            irq,
        })
    }

    /// Base guest physical address of region 0, if mapped.
    pub fn base_gpa(&self) -> Option<u64> {
        self.regions.first().map(|r| r.gpa)
    }
}

impl std::fmt::Debug for VfioPlatformDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VfioPlatformDevice")
            .field("id", &self.id)
            .field(
                "regions",
                &self
                    .regions
                    .iter()
                    .map(|r| (r.slot, r.gpa, r.mapping.size()))
                    .collect::<Vec<_>>(),
            )
            .field("irq", &self.irq.as_ref().map(|i| i.gsi))
            .finish()
    }
}

/// Debug probe for the RK3588 NPU core 2 passthrough setup: open both
/// noiommu groups, mmap the core's region 0 (pc window) and read the
/// REG_PC_VERSION / REG_PC_VERSION_NUM registers.
pub fn rknn_probe() -> Result<()> {
    let (_core_container, core_group) = open_group(Path::new("/dev/vfio/noiommu-15"))?;
    let core_device = get_device_fd(&core_group, "fdad0000.npu")?;
    let (_mmu_container, mmu_group) = open_group(Path::new("/dev/vfio/noiommu-16"))?;
    let mmu_device = get_device_fd(&mmu_group, "fdada000.iommu")?;

    let (core_regions, core_irqs) = get_device_info(&core_device)?;
    println!(
        "fdad0000.npu: {} region(s), {} irq(s)",
        core_regions.len(),
        core_irqs.len()
    );
    for r in &core_regions {
        println!(
            "  region {}: size {:#x}, offset {:#x}, flags {:#x}",
            r.index, r.size, r.offset, r.flags
        );
    }
    for i in &core_irqs {
        println!("  irq {}: count {}, flags {:#x}", i.index, i.count, i.flags);
    }

    let (mmu_regions, mmu_irqs) = get_device_info(&mmu_device)?;
    println!(
        "fdada000.iommu: {} region(s), {} irq(s)",
        mmu_regions.len(),
        mmu_irqs.len()
    );
    for r in &mmu_regions {
        println!(
            "  region {}: size {:#x}, offset {:#x}, flags {:#x}",
            r.index, r.size, r.offset, r.flags
        );
    }

    let region0 = &core_regions[0];
    let mapping = mmap_region(&core_device, region0, region0.size as usize)?;
    let base = mapping.as_ptr() as *const u32;
    // SAFETY: `mapping` covers the whole pc window; offsets 0x0 and 0x4 are
    // within bounds. Volatile reads since this is device MMIO.
    let (version, version_num) = unsafe {
        (
            std::ptr::read_volatile(base),
            std::ptr::read_volatile(base.add(1)),
        )
    };
    println!("REG_PC_VERSION     = {version:#010x}");
    println!("REG_PC_VERSION_NUM = {version_num:#010x}");

    // REG_PC_VERSION is the NPU signature register (0x46495245 = "FIRE" on
    // RK3588; matches what the host rocket driver read at boot). It must be
    // neither 0 (device off) nor all-ones (bus error fill).
    // REG_PC_VERSION_NUM reads as 0 on this silicon (the host rocket driver
    // adds `VERSION_NUM & 0xffff` to the version, and its boot-time print
    // equals the plain VERSION value), so it is informational only.
    if version == 0 || version == 0xffff_ffff {
        eprintln!("rknn_probe: FAIL (REG_PC_VERSION invalid; is the keeper module loaded?)");
        return Err(PlatformDeviceError::MmapRegion(io::Error::other(
            "invalid REG_PC_VERSION value",
        )));
    }
    println!("rknn_probe: OK");
    Ok(())
}
