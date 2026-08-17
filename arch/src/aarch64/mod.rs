// Copyright 2020 Arm Limited (or its affiliates). All rights reserved.
// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/// Module for cache info.
pub mod cache;
/// Module for the flattened device tree.
pub mod fdt;
/// Layout for this aarch64 system.
pub mod layout;
/// Module for loading UEFI binary.
pub mod uefi;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::sync::{Arc, Mutex};

use hypervisor::arch::aarch64::gic::Vgic;
use hypervisor::arch::aarch64::regs::MPIDR_EL1;
use log::{Level, log_enabled};
use thiserror::Error;
use vm_memory::{
    Address, GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryAtomic, GuestMemoryRegion,
};

pub use self::fdt::DeviceInfoForFdt;
use crate::{DeviceType, GuestMemoryMmap, NumaNodes, PciSpaceInfo, RegionType};

pub const _NSIG: i32 = 65;

/// Errors thrown while configuring aarch64 system.
#[derive(Debug, Error)]
pub enum Error {
    /// Failed to create a FDT.
    #[error("Failed to create a FDT")]
    SetupFdt,

    /// Failed to write FDT to memory.
    #[error("Failed to write FDT to memory")]
    WriteFdtToMemory(#[source] fdt::Error),

    /// Failed to create a GIC.
    #[error("Failed to create a GIC")]
    SetupGic,

    /// Failed to compute the initramfs address.
    #[error("Failed to compute the initramfs address")]
    InitramfsAddress,

    /// Error configuring the general purpose registers
    #[error("Error configuring the general purpose registers")]
    RegsConfiguration(#[source] hypervisor::HypervisorCpuError),

    /// Error configuring the MPIDR register
    #[error("Error configuring the MPIDR register")]
    VcpuRegMpidr(#[source] hypervisor::HypervisorCpuError),

    /// Error initializing PMU for vcpu
    #[error("Error initializing PMU for vcpu")]
    VcpuInitPmu,
}

#[derive(Debug, Copy, Clone)]
/// Specifies the entry point address where the guest must start
/// executing code.
pub struct EntryPoint {
    /// Address in guest memory where the guest must start execution
    pub entry_addr: GuestAddress,
}

/// Configure the specified VCPU, and return its MPIDR.
pub fn configure_vcpu(
    vcpu: &dyn hypervisor::Vcpu,
    id: u32,
    boot_setup: Option<(EntryPoint, &GuestMemoryAtomic<GuestMemoryMmap>)>,
) -> super::Result<u64> {
    if let Some((kernel_entry_point, guest_memory)) = boot_setup {
        // The FDT is placed at the beginning of the first RAM region, which
        // is not necessarily `layout::RAM_START` when the guest RAM is
        // identity-mapped onto a host physical carve-out.
        let fdt_addr = first_ram_start(&guest_memory.memory())
            .unchecked_add(layout::FDT_START.0 - layout::RAM_START.0);
        vcpu.setup_regs(
            id,
            kernel_entry_point.entry_addr.raw_value(),
            fdt_addr.raw_value(),
        )
        .map_err(Error::RegsConfiguration)?;
    }

    let mpidr = vcpu.get_sys_reg(MPIDR_EL1).map_err(Error::VcpuRegMpidr)?;
    Ok(mpidr)
}

/// Returns the start address of the first guest RAM region. This is
/// `layout::RAM_START` with the standard memory layout, and the
/// identity-map base (host physical address of the carve-out) when the
/// guest RAM is identity-mapped.
pub fn first_ram_start(guest_mem: &GuestMemoryMmap) -> GuestAddress {
    guest_mem
        .iter()
        .next()
        .map(GuestMemoryRegion::start_addr)
        .expect("GuestMemory must have at least one memory region")
}

/// Returns the guest address where the kernel image should be loaded: the
/// start of the first RAM region plus the standard offset
/// (`KERNEL_START - RAM_START`), aligned up to 2 MiB.
pub fn kernel_load_addr(guest_mem: &GuestMemoryMmap) -> GuestAddress {
    const ALIGNMENT: u64 = 0x20_0000;
    let kernel_offset = layout::KERNEL_START.0 - layout::RAM_START.0;
    let addr = first_ram_start(guest_mem).raw_value() + kernel_offset;
    GuestAddress(addr.div_ceil(ALIGNMENT) * ALIGNMENT)
}

/// Returns the guest address where the ACPI tables (RSDP) should be placed:
/// the start of the first RAM region plus the standard offset
/// (`ACPI_START - RAM_START`).
pub fn acpi_load_addr(guest_mem: &GuestMemoryMmap) -> GuestAddress {
    first_ram_start(guest_mem).unchecked_add(layout::ACPI_START.0 - layout::RAM_START.0)
}

pub fn arch_memory_regions() -> Vec<(GuestAddress, usize, RegionType)> {
    vec![
        // 0 MiB ~ 256 MiB: UEFI, GIC and legacy devices
        (
            GuestAddress(0),
            layout::MEM_32BIT_DEVICES_START.0 as usize,
            RegionType::Reserved,
        ),
        // 256 MiB ~ 768 MiB: MMIO space
        (
            layout::MEM_32BIT_DEVICES_START,
            layout::MEM_32BIT_DEVICES_SIZE as usize,
            RegionType::SubRegion,
        ),
        // 768 MiB ~ 1 GiB: reserved. The leading 256M for PCIe MMCONFIG space
        (
            layout::PCI_MMCONFIG_START,
            layout::PCI_MMCONFIG_SIZE as usize,
            RegionType::Reserved,
        ),
        // 1GiB ~ 4032 MiB: RAM before the gap
        (
            layout::RAM_START,
            layout::MEM_32BIT_RESERVED_START.unchecked_offset_from(layout::RAM_START) as usize,
            RegionType::Ram,
        ),
        // 4GiB ~ inf: RAM after the gap
        (layout::RAM_64BIT_START, usize::MAX, RegionType::Ram),
        // Add the 32-bit reserved memory hole as a reserved region
        (
            layout::MEM_32BIT_RESERVED_START,
            layout::MEM_32BIT_RESERVED_SIZE as usize,
            RegionType::Reserved,
        ),
    ]
}

/// Configures the system and should be called once per vm before starting vcpu threads.
#[expect(clippy::too_many_arguments)]
pub fn configure_system<T: DeviceInfoForFdt + Clone + Debug, S: BuildHasher>(
    guest_mem: &GuestMemoryMmap,
    cmdline: &str,
    vcpu_mpidr: &[u64],
    vcpu_topology: Option<(u16, u16, u16, u16)>,
    device_info: &HashMap<(DeviceType, String), T, S>,
    initrd: &Option<super::InitramfsConfig>,
    pci_space_info: &[PciSpaceInfo],
    virtio_iommu_bdf: Option<u32>,
    gic_device: &Arc<Mutex<dyn Vgic>>,
    numa_nodes: &NumaNodes,
    pmu_supported: bool,
) -> super::Result<()> {
    let fdt_final = fdt::create_fdt(
        guest_mem,
        cmdline,
        vcpu_mpidr,
        vcpu_topology,
        device_info,
        gic_device,
        initrd,
        pci_space_info,
        numa_nodes,
        virtio_iommu_bdf,
        pmu_supported,
    )
    .map_err(|_| Error::SetupFdt)?;

    if log_enabled!(Level::Debug) {
        fdt::print_fdt(&fdt_final);
    }

    fdt::write_fdt_to_memory(&fdt_final, guest_mem).map_err(Error::WriteFdtToMemory)?;

    Ok(())
}

/// Returns the memory address where the initramfs could be loaded.
pub fn initramfs_load_addr(
    guest_mem: &GuestMemoryMmap,
    initramfs_size: usize,
) -> super::Result<u64> {
    let round_to_pagesize = |size| (size + (super::PAGE_SIZE - 1)) & !(super::PAGE_SIZE - 1);
    match guest_mem
        .last_addr()
        .checked_sub(round_to_pagesize(initramfs_size) as u64 - 1)
    {
        Some(offset) => {
            if guest_mem.address_in_range(offset) {
                Ok(offset.raw_value())
            } else {
                Err(super::Error::PlatformSpecific(Error::InitramfsAddress))
            }
        }
        None => Err(super::Error::PlatformSpecific(Error::InitramfsAddress)),
    }
}

pub fn get_host_cpu_phys_bits(hypervisor: &dyn hypervisor::Hypervisor) -> u8 {
    let host_cpu_phys_bits = hypervisor.get_host_ipa_limit().try_into().unwrap();
    if host_cpu_phys_bits == 0 {
        // Host kernel does not support `get_host_ipa_limit`,
        // we return the default value 40 here.
        40
    } else {
        host_cpu_phys_bits
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_arch_memory_regions_dram() {
        let regions = arch_memory_regions();
        assert_eq!(6, regions.len());
        assert_eq!(layout::RAM_START, regions[3].0);
        assert_eq!(RegionType::Ram, regions[3].2);
        assert_eq!(RegionType::Reserved, regions[5].2);
        assert_eq!(RegionType::Ram, regions[4].2);
    }
}
