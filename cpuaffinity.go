package rknnlite

import (
	"fmt"
	"syscall"
	"unsafe"
)

const (
	// RK3588FastCores is the cpu affinity mask of the fast cortex A76 cores 4-7
	RK3588FastCores = uintptr(0b11110000)
	// RK3588SlowCores is the cpu affinity mask of the efficient cortex A55 cores 0-3
	RK3588SlowCores = uintptr(0b00001111)
	// RK3588Allcores is the cpu affinity mask for all cortext A76 and A55 cores 0-7
	RK3588AllCores = uintptr(0b11111111)

	// RK3582FastCores is the cpu affinity mask of the fast cortex A76 cores 4-5
	RK3582FastCores = uintptr(0b00110000)
	// RK3582SlowCores is the cpu affinity mask of the efficient cortex A55 cores 0-3
	RK3582SlowCores = uintptr(0b00001111)
	// RK3582Allcores is the cpu affinity mask for all cortext A76 and A55 cores 0-5
	RK3582AllCores = uintptr(0b00111111)
)

// SetCPUAffinity sets the CPU Affinity mask of the program to run on the specified
// cores
func SetCPUAffinity(mask uintptr) error {

	_, _, err := syscall.RawSyscall(syscall.SYS_SCHED_SETAFFINITY, 0,
		unsafe.Sizeof(mask), uintptr(unsafe.Pointer(&mask)))

	if err != 0 {
		return fmt.Errorf("failed to set CPU affinity: %w", err)
	}

	return nil
}

// GetCPUAffinity gets the current CPU Affinity mask the program is running on
func GetCPUAffinity() (uintptr, error) {

	var mask uintptr

	_, _, err := syscall.RawSyscall(syscall.SYS_SCHED_GETAFFINITY, 0,
		unsafe.Sizeof(mask), uintptr(unsafe.Pointer(&mask)))

	if err != 0 {
		return 0, fmt.Errorf("failed to get CPU affinity: %w", err)
	}

	return mask, nil
}

// CPUCoreMask calculates the core mask by passing in the CPU core numbers as a
// slice, eg: []int{4,5,6,7}
func CPUCoreMask(cores []int) uintptr {

	var mask uintptr

	for _, core := range cores {
		mask |= 1 << core
	}

	return mask
}
