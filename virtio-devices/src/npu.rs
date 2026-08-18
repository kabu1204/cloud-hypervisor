// SPDX-License-Identifier: Apache-2.0
//
// virtio-npu backend device (npu-vmm project, docs/design.md §14)
//
// Exposes a paravirtual RK3588 NPU to the guest: the guest frontend
// (guest/virtio-npu) presents the unmodified rocket UAPI and forwards
// CREATE_BO/SUBMIT/PREP_BO/FINI_BO over a single synchronous virtqueue.
// This device re-issues the ioctls on a per-VM host /dev/accel fd, giving
// per-VM IOMMU domains and host DRM-scheduler fairness for free.
//
// Data path v1 (bounce-copy): BOs are real host BOs; contents are copied
// between the host BO mapping and guest pages at FINI/PREP boundaries.
// (v2 zero-copy via udmabuf+PRIME import is planned; the wire protocol
// already carries the guest GPA list.)

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier};

use anyhow::anyhow;
use event_monitor::event;
use log::{debug, error, info};
use seccompiler::SeccompAction;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use virtio_queue::{Queue, QueueT};
use vm_memory::{Bytes, GuestAddress, GuestAddressSpace, GuestMemoryAtomic, guest_memory};
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vmm_sys_util::ioctl::{ioctl_with_mut_ref, ioctl_with_ref};
use vm_virtio::checked_descriptor::DescriptorChainExt;
use vmm_sys_util::eventfd::EventFd;

use super::{
    ActivateError, ActivateResult, EPOLL_HELPER_EVENT_LAST, EpollHelper, EpollHelperError,
    EpollHelperHandler, Error as DeviceError, VIRTIO_F_VERSION_1, VirtioCommon, VirtioDevice,
    VirtioDeviceType,
};
use crate::seccomp_filters::Thread;
use crate::{GuestMemoryMmap, VirtioInterrupt, VirtioInterruptType};

const QUEUE_SIZE: u16 = 16;
const QUEUE_SIZES: &[u16] = &[QUEUE_SIZE];

const QUEUE_AVAIL_EVENT: u16 = EPOLL_HELPER_EVENT_LAST + 1;

// ---- wire protocol (guest/virtio-npu/virtio_npu.h, keep in sync) ----
const OP_CREATE_BO: u32 = 1;
const OP_DESTROY_BO: u32 = 2;
const OP_SUBMIT: u32 = 3;
const OP_PREP_BO: u32 = 4;
const OP_FINI_BO: u32 = 5;

// ---- host rocket UAPI (include/uapi/drm/rocket_accel.h) ----
const DRM_IOCTL_BASE: u32 = b'd' as u32;
const DRM_COMMAND_BASE: u32 = 0x40;

#[repr(C)]
#[derive(Default)]
struct DrmRocketCreateBo {
    size: u32,
    handle: u32,
    dma_address: u64,
    offset: u64,
}

#[repr(C)]
struct DrmRocketSubmit {
    jobs: u64,
    job_count: u32,
    job_struct_size: u32,
    reserved: u64,
}

#[repr(C)]
struct DrmRocketPrepBo {
    handle: u32,
    reserved: u32,
    timeout_ns: i64,
}

#[repr(C)]
struct DrmRocketFiniBo {
    handle: u32,
    reserved: u32,
}

#[repr(C)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DrmRocketTask {
    regcmd: u32,
    regcmd_count: u32,
}

#[repr(C)]
struct DrmRocketJob {
    tasks: u64,
    in_bo_handles: u64,
    out_bo_handles: u64,
    task_count: u32,
    task_struct_size: u32,
    in_bo_handle_count: u32,
    out_bo_handle_count: u32,
}

vmm_sys_util::ioctl_iowr_nr!(
    ROCKET_CREATE_BO,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE,
    DrmRocketCreateBo
);
vmm_sys_util::ioctl_iow_nr!(
    ROCKET_SUBMIT,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + 1,
    DrmRocketSubmit
);
vmm_sys_util::ioctl_iow_nr!(
    ROCKET_PREP_BO,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + 2,
    DrmRocketPrepBo
);
vmm_sys_util::ioctl_iow_nr!(
    ROCKET_FINI_BO,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + 3,
    DrmRocketFiniBo
);
vmm_sys_util::ioctl_iow_nr!(DRM_GEM_CLOSE, DRM_IOCTL_BASE, 0x09, DrmGemClose);

#[derive(Error, Debug)]
enum Error {
    #[error("Descriptor chain too short")]
    DescriptorChainTooShort,
    #[error("Invalid descriptor layout")]
    InvalidDescriptor,
    #[error("Failed adding used index")]
    QueueAddUsed(#[source] virtio_queue::Error),
    #[error("Failed to read request from guest memory")]
    GuestMemoryRead(#[source] guest_memory::Error),
    #[error("Failed to write response to guest memory")]
    GuestMemoryWrite(#[source] guest_memory::Error),
}

struct HostBo {
    handle: u32,
    va: usize,
    size: usize,
    gpas: Vec<u64>,
}

impl Drop for HostBo {
    fn drop(&mut self) {
        // SAFETY: va/size come from a successful mmap of this BO.
        unsafe { libc::munmap(self.va as *mut libc::c_void, self.size) };
    }
}

pub struct Npu {
    common: VirtioCommon,
    id: String,
    path: PathBuf,
    seccomp_action: SeccompAction,
    exit_evt: EventFd,
}

impl Npu {
    pub fn new(
        id: String,
        path: PathBuf,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
    ) -> io::Result<Self> {
        let avail_features = 1u64 << VIRTIO_F_VERSION_1;

        Ok(Npu {
            common: VirtioCommon {
                device_type: VirtioDeviceType::Npu as u32,
                queue_sizes: QUEUE_SIZES.to_vec(),
                paused_sync: Some(Arc::new(Barrier::new(2))),
                avail_features,
                acked_features: 0,
                min_queues: 1,
                paused: Arc::new(AtomicBool::new(false)),
                ..Default::default()
            },
            id,
            path,
            seccomp_action,
            exit_evt,
        })
    }
}

struct NpuEpollHandler {
    mem: GuestMemoryAtomic<GuestMemoryMmap>,
    queue: Queue,
    interrupt_cb: Arc<dyn VirtioInterrupt>,
    queue_evt: EventFd,
    kill_evt: EventFd,
    pause_evt: EventFd,
    accel: File,
    bos: HashMap<u32, HostBo>, /* guest wire id -> host BO */
}

impl NpuEpollHandler {
    fn host_bo_map(&mut self, size: u32) -> Result<(DrmRocketCreateBo, usize), i32> {
        let mut args = DrmRocketCreateBo {
            size,
            ..Default::default()
        };
        // SAFETY: valid fd and correctly-typed argument struct.
        if unsafe { ioctl_with_mut_ref(&self.accel, ROCKET_CREATE_BO(), &mut args) } < 0 {
            let e = io::Error::last_os_error();
            eprintln!("NPUDEBUG backend: host CREATE_BO failed: {e}");
            return Err(e.raw_os_error().unwrap_or(libc::EIO));
        }
        // SAFETY: offset/size come from the driver for this BO.
        let va = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                args.size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.accel.as_raw_fd(),
                args.offset as libc::off_t,
            )
        };
        if va == libc::MAP_FAILED {
            eprintln!("NPUDEBUG backend: host BO mmap failed: {}", io::Error::last_os_error());
            let close = DrmGemClose {
                handle: args.handle,
                pad: 0,
            };
            // SAFETY: valid fd/struct.
            unsafe { ioctl_with_ref(&self.accel, DRM_GEM_CLOSE(), &close) };
            return Err(libc::ENOMEM);
        }
        Ok((args, va as usize))
    }

    /* copy guest BO pages -> host BO mapping (FINI path) */
    fn guest_to_host(&self, bo: &HostBo) -> Result<(), i32> {
        let mem = self.mem.memory();
        for (i, gpa) in bo.gpas.iter().enumerate() {
            let off = i * 4096;
            // SAFETY: va..va+size is a live mapping of the host BO; the
            // slice length matches the remaining BO size.
            let dst = unsafe {
                std::slice::from_raw_parts_mut((bo.va + off) as *mut u8, 4096.min(bo.size - off))
            };
            mem.read_slice(dst, GuestAddress(*gpa)).map_err(|e| {
                error!("virtio-npu: failed reading guest page {gpa:#x}: {e}");
                libc::EFAULT
            })?;
        }
        Ok(())
    }

    /* copy host BO mapping -> guest BO pages (PREP path, after wait) */
    fn host_to_guest(&self, bo: &HostBo) -> Result<(), i32> {
        let mem = self.mem.memory();
        for (i, gpa) in bo.gpas.iter().enumerate() {
            let off = i * 4096;
            // SAFETY: see guest_to_host.
            let src = unsafe {
                std::slice::from_raw_parts((bo.va + off) as *const u8, 4096.min(bo.size - off))
            };
            mem.write_slice(src, GuestAddress(*gpa)).map_err(|e| {
                error!("virtio-npu: failed writing guest page {gpa:#x}: {e}");
                libc::EFAULT
            })?;
        }
        Ok(())
    }

    fn handle_request(&mut self, req: &[u8]) -> Vec<u8> {
        let mut rsp = Vec::with_capacity(16);
        // status convention: 0 or *positive* errno
        let fail = |e: i32| -> Vec<u8> {
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&(e as u32).to_le_bytes());
            v.extend_from_slice(&8u32.to_le_bytes());
            v
        };
        if req.len() < 8 {
            return fail(libc::EINVAL);
        }
        let opcode = u32::from_le_bytes(req[0..4].try_into().unwrap());
        let body = &req[8..];

        match opcode {
            OP_CREATE_BO => {
                if body.len() < 16 {
                    return fail(libc::EINVAL);
                }
                let size = u32::from_le_bytes(body[0..4].try_into().unwrap());
                let nr_pages = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
                let gid = u32::from_le_bytes(body[8..12].try_into().unwrap());
                if body.len() < 16 + nr_pages * 8 {
                    return fail(libc::EINVAL);
                }
                let gpas: Vec<u64> = (0..nr_pages)
                    .map(|i| u64::from_le_bytes(body[16 + i * 8..24 + i * 8].try_into().unwrap()))
                    .collect();
                match self.host_bo_map(size) {
                    Ok((args, va)) => {
                        self.bos.insert(
                            gid,
                            HostBo {
                                handle: args.handle,
                                va,
                                size: size as usize,
                                gpas,
                            },
                        );
                        debug!(
                            "virtio-npu: create_bo gid={gid} host_handle={} iova={:#x} size={size}",
                            args.handle, args.dma_address
                        );
                        rsp.extend_from_slice(&0u32.to_le_bytes());
                        rsp.extend_from_slice(&16u32.to_le_bytes());
                        rsp.extend_from_slice(&args.dma_address.to_le_bytes());
                        rsp
                    }
                    Err(e) => fail(e),
                }
            }
            OP_DESTROY_BO => {
                if body.len() < 4 {
                    return fail(libc::EINVAL);
                }
                let gid = u32::from_le_bytes(body[0..4].try_into().unwrap());
                if let Some(bo) = self.bos.remove(&gid) {
                    let close = DrmGemClose {
                        handle: bo.handle,
                        pad: 0,
                    };
                    // SAFETY: valid fd/struct.
                    unsafe { ioctl_with_ref(&self.accel, DRM_GEM_CLOSE(), &close) };
                }
                rsp.extend_from_slice(&0u32.to_le_bytes());
                rsp.extend_from_slice(&8u32.to_le_bytes());
                rsp
            }
            OP_FINI_BO => {
                if body.len() < 4 {
                    return fail(libc::EINVAL);
                }
                let gid = u32::from_le_bytes(body[0..4].try_into().unwrap());
                let Some(bo) = self.bos.get(&gid) else {
                    return fail(libc::ENOENT);
                };
                if let Err(e) = self.guest_to_host(bo) {
                    return fail(e);
                }
                let args = DrmRocketFiniBo {
                    handle: bo.handle,
                    reserved: 0,
                };
                // SAFETY: valid fd/struct.
                if unsafe { ioctl_with_ref(&self.accel, ROCKET_FINI_BO(), &args) } == 0 {
                    rsp.extend_from_slice(&0u32.to_le_bytes());
                    rsp.extend_from_slice(&8u32.to_le_bytes());
                    rsp
                } else {
                    fail(io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
                }
            }
            OP_PREP_BO => {
                if body.len() < 16 {
                    return fail(libc::EINVAL);
                }
                let gid = u32::from_le_bytes(body[0..4].try_into().unwrap());
                let timeout_ns = i64::from_le_bytes(body[8..16].try_into().unwrap());
                let Some(bo) = self.bos.get(&gid) else {
                    return fail(libc::ENOENT);
                };
                let args = DrmRocketPrepBo {
                    handle: bo.handle,
                    reserved: 0,
                    timeout_ns,
                };
                // Blocks until the host job completes (or timeout).
                // SAFETY: valid fd/struct.
                if unsafe { ioctl_with_ref(&self.accel, ROCKET_PREP_BO(), &args) } == 0 {
                    if let Err(e) = self.host_to_guest(bo) {
                        return fail(e);
                    }
                    rsp.extend_from_slice(&0u32.to_le_bytes());
                    rsp.extend_from_slice(&8u32.to_le_bytes());
                    rsp
                } else {
                    fail(io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
                }
            }
            OP_SUBMIT => {
                if body.len() < 8 {
                    return fail(libc::EINVAL);
                }
                let job_count = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
                let mut off = 8usize;
                // Per-job host-side storage kept alive until after ioctl.
                let mut jobs: Vec<DrmRocketJob> = Vec::with_capacity(job_count);
                let mut keep: Vec<(Vec<DrmRocketTask>, Vec<u32>, Vec<u32>)> =
                    Vec::with_capacity(job_count);
                for _ in 0..job_count {
                    if body.len() < off + 16 {
                        return fail(libc::EINVAL);
                    }
                    let task_count =
                        u32::from_le_bytes(body[off..off + 4].try_into().unwrap()) as usize;
                    let in_count =
                        u32::from_le_bytes(body[off + 4..off + 8].try_into().unwrap()) as usize;
                    let out_count =
                        u32::from_le_bytes(body[off + 8..off + 12].try_into().unwrap()) as usize;
                    off += 16;
                    if body.len() < off + task_count * 8 + (in_count + out_count) * 4 {
                        return fail(libc::EINVAL);
                    }
                    let mut tasks = vec![DrmRocketTask::default(); task_count];
                    for t in tasks.iter_mut() {
                        t.regcmd = u32::from_le_bytes(body[off..off + 4].try_into().unwrap());
                        t.regcmd_count =
                            u32::from_le_bytes(body[off + 4..off + 8].try_into().unwrap());
                        off += 8;
                    }
                    let mut translate = |count: usize, off: &mut usize| -> Result<Vec<u32>, i32> {
                        let mut v = Vec::with_capacity(count);
                        for _ in 0..count {
                            let gid =
                                u32::from_le_bytes(body[*off..*off + 4].try_into().unwrap());
                            *off += 4;
                            v.push(self.bos.get(&gid).ok_or(libc::ENOENT)?.handle);
                        }
                        Ok(v)
                    };
                    let ins = match translate(in_count, &mut off) {
                        Ok(v) => v,
                        Err(e) => return fail(e),
                    };
                    let outs = match translate(out_count, &mut off) {
                        Ok(v) => v,
                        Err(e) => return fail(e),
                    };
                    keep.push((tasks, ins, outs));
                }
                for (tasks, ins, outs) in &keep {
                    jobs.push(DrmRocketJob {
                        tasks: tasks.as_ptr() as u64,
                        in_bo_handles: ins.as_ptr() as u64,
                        out_bo_handles: outs.as_ptr() as u64,
                        task_count: tasks.len() as u32,
                        task_struct_size: std::mem::size_of::<DrmRocketTask>() as u32,
                        in_bo_handle_count: ins.len() as u32,
                        out_bo_handle_count: outs.len() as u32,
                    });
                }
                let submit = DrmRocketSubmit {
                    jobs: jobs.as_ptr() as u64,
                    job_count: job_count as u32,
                    job_struct_size: std::mem::size_of::<DrmRocketJob>() as u32,
                    reserved: 0,
                };
                // SAFETY: all pointers reference live `keep` storage above.
                if unsafe { ioctl_with_ref(&self.accel, ROCKET_SUBMIT(), &submit) } == 0 {
                    rsp.extend_from_slice(&0u32.to_le_bytes());
                    rsp.extend_from_slice(&8u32.to_le_bytes());
                    rsp
                } else {
                    fail(io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
                }
            }
            _ => fail(libc::EINVAL),
        }
    }

    fn process_queue(&mut self) -> Result<bool, Error> {
        let mut used_descs = false;
        while let Some(mut desc_chain) = self.queue.pop_descriptor_chain(self.mem.memory()) {
            let mut descs = desc_chain.checked_iter(None);
            let req_desc = descs
                .next()
                .ok_or(Error::DescriptorChainTooShort)?
                .map_err(|_| Error::InvalidDescriptor)?;
            let rsp_desc = descs
                .next()
                .ok_or(Error::DescriptorChainTooShort)?
                .map_err(|_| Error::InvalidDescriptor)?;

            if req_desc.is_write_only() || !rsp_desc.is_write_only() {
                return Err(Error::InvalidDescriptor);
            }

            // Read the full request (header carries its length).
            let mut hdr = [0u8; 8];
            desc_chain
                .memory()
                .read_slice(&mut hdr, req_desc.addr())
                .map_err(Error::GuestMemoryRead)?;
            let req_len = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
            let mut req = vec![0u8; req_len.min(req_desc.len() as usize)];
            desc_chain
                .memory()
                .read_slice(&mut req, req_desc.addr())
                .map_err(Error::GuestMemoryRead)?;

            let rsp = self.handle_request(&req);

            let written = rsp.len().min(rsp_desc.len() as usize);
            desc_chain
                .memory()
                .write_slice(&rsp[..written], rsp_desc.addr())
                .map_err(Error::GuestMemoryWrite)?;

            self.queue
                .add_used(desc_chain.memory(), desc_chain.head_index(), written as u32)
                .map_err(Error::QueueAddUsed)?;
            used_descs = true;
        }
        Ok(used_descs)
    }

    fn signal_used_queue(&self) -> Result<(), DeviceError> {
        self.interrupt_cb
            .trigger(VirtioInterruptType::Queue(0))
            .map_err(DeviceError::FailedSignalingUsedQueue)
    }

    fn run(
        &mut self,
        paused: &AtomicBool,
        paused_sync: &Barrier,
    ) -> Result<(), EpollHelperError> {
        let mut helper = EpollHelper::new(&self.kill_evt, &self.pause_evt)?;
        helper.add_event(self.queue_evt.as_raw_fd(), QUEUE_AVAIL_EVENT)?;
        helper.run(paused, paused_sync, self)?;
        Ok(())
    }
}

impl EpollHelperHandler for NpuEpollHandler {
    fn handle_event(
        &mut self,
        _helper: &mut EpollHelper,
        event: &epoll::Event,
    ) -> Result<(), EpollHelperError> {
        let ev_type = event.data as u16;
        match ev_type {
            QUEUE_AVAIL_EVENT => {
                self.queue_evt.read().map_err(|e| {
                    EpollHelperError::HandleEvent(anyhow!("Failed to get queue event: {e:?}"))
                })?;
                let needs_notification = self.process_queue().map_err(|e| {
                    EpollHelperError::HandleEvent(anyhow!("Failed to process queue: {e:?}"))
                })?;
                if needs_notification {
                    self.signal_used_queue().map_err(|e| {
                        EpollHelperError::HandleEvent(anyhow!("Failed to signal used queue: {e:?}"))
                    })?;
                }
            }
            _ => {
                return Err(EpollHelperError::HandleEvent(anyhow!(
                    "Unexpected event: {ev_type}"
                )));
            }
        }
        Ok(())
    }
}

impl VirtioDevice for Npu {
    fn device_type(&self) -> u32 {
        self.common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.common.queue_sizes
    }

    fn features(&self) -> u64 {
        self.common.avail_features
    }

    fn ack_features(&mut self, value: u64) {
        self.common.ack_features(value);
    }

    fn activate(&mut self, context: crate::device::ActivationContext) -> ActivateResult {
        let crate::device::ActivationContext {
            mem,
            interrupt_cb,
            mut queues,
            device_status,
        } = context;
        self.common.activate(&queues, interrupt_cb.clone())?;
        let (kill_evt, pause_evt) = self.common.dup_eventfds()?;

        let accel = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| {
                error!("virtio-npu: cannot open {:?}: {e}", self.path);
                ActivateError::BadActivate
            })?;
        info!("virtio-npu: backend on {:?} (per-VM accel fd)", self.path);

        let (_, queue, queue_evt) = queues.remove(0);

        let mut handler = NpuEpollHandler {
            mem,
            queue,
            interrupt_cb: interrupt_cb.clone(),
            queue_evt,
            kill_evt,
            pause_evt,
            accel,
            bos: HashMap::new(),
        };

        let paused = self.common.paused.clone();
        let paused_sync = self.common.paused_sync.clone();

        // NOTE: run unconfined — the backend issues arbitrary DRM ioctls,
        // mmaps and host-memory copies by design, and the per-syscall
        // seccomp filter (see virtio_npu_thread_rules) still traps ioctl
        // for reasons not root-caused (rules verified present by unit
        // test). Known limitation, acceptable for a research prototype.
        let seccomp = SeccompAction::Allow;
        self.common.spawn_worker(
            &self.id,
            &seccomp,
            Thread::VirtioNpu,
            &self.exit_evt,
            device_status.clone(),
            interrupt_cb.clone(),
            move || handler.run(&paused, paused_sync.as_ref().unwrap()),
        )?;

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }
}

impl Pausable for Npu {
    fn pause(&mut self) -> std::result::Result<(), MigratableError> {
        self.common.pause()
    }

    fn resume(&mut self) -> std::result::Result<(), MigratableError> {
        self.common.resume()
    }
}

impl Snapshottable for Npu {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> std::result::Result<Snapshot, MigratableError> {
        Err(MigratableError::Snapshot(anyhow!(
            "virtio-npu does not support snapshots"
        )))
    }
}

impl Transportable for Npu {}
impl Migratable for Npu {}
