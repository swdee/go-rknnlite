package rknnlite

import (
	"fmt"
	"strings"
	"syscall"
	"unsafe"
)

const (
	// RK3588FastCores is the cpu affinity mask of the fast cortex A76 cores 4-7
	RK3588FastCores = uintptr(0b11110000)
	// RK3588SlowCores is the cpu affinity mask of the efficient cortex A55 cores 0-3
	RK3588SlowCores = uintptr(0b00001111)
	// RK3588Allcores is the cpu affinity mask for all cortex A76 and A55 cores 0-7
	RK3588AllCores = uintptr(0b11111111)

	// RK3582FastCores is the cpu affinity mask of the fast cortex A76 cores 4-5
	RK3582FastCores = uintptr(0b00110000)
	// RK3582SlowCores is the cpu affinity mask of the efficient cortex A55 cores 0-3
	RK3582SlowCores = uintptr(0b00001111)
	// RK3582Allcores is the cpu affinity mask for all cortex A76 and A55 cores 0-5
	RK3582AllCores = uintptr(0b00111111)

	// RK3576FastCores is the cpu affinity mask of the fast cortex A72 cores 4-7
	RK3576FastCores = uintptr(0b11110000)
	// RK3576SlowCores is the cpu affinity mask of the efficient cortex A53 cores 0-3
	RK3576SlowCores = uintptr(0b00001111)
	// RK3576Allcores is the cpu affinity mask for all cortex A72 and A53 cores 0-7
	RK3576AllCores = uintptr(0b11111111)

	// RK3568AllCores is the cpu affinity mask of all cortex A55 (2Ghz) cores 0-3
	RK3568AllCores = uintptr(0b00001111)

	// RK3566AllCores is the cpu affinity mask of all cortex A55 (1.6Ghz) cores 0-3
	RK3566AllCores = uintptr(0b00001111)

	// RK3562AllCores is the cpu affinity mask of all cortex A53 cores 0-3
	RK3562AllCores = uintptr(0b00001111)
)

// CoreType specifies the CPU core type
type CoreType int

const (
	FastCores CoreType = 0
	SlowCores CoreType = 1
	AllCores  CoreType = 2
)

// coreMaskList defines a list of CPU core masks for lookup by key
var coreMaskList = map[string]map[CoreType]uintptr{
	"rk3562": {
		SlowCores: RK3562AllCores,
		FastCores: RK3562AllCores,
		AllCores:  RK3562AllCores,
	},
	"rk3566": {
		SlowCores: RK3566AllCores,
		FastCores: RK3566AllCores,
		AllCores:  RK3566AllCores,
	},
	"rk3568": {
		SlowCores: RK3568AllCores,
		FastCores: RK3568AllCores,
		AllCores:  RK3568AllCores,
	},
	"rk3576": {
		SlowCores: RK3576SlowCores,
		FastCores: RK3576FastCores,
		AllCores:  RK3576AllCores,
	},
	"rk3582": {
		SlowCores: RK3582SlowCores,
		FastCores: RK3582FastCores,
		AllCores:  RK3582AllCores,
	},
	"rk3588": {
		SlowCores: RK3588SlowCores,
		FastCores: RK3588FastCores,
		AllCores:  RK3588AllCores,
	},
}

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

// SetCPUAffinityByPlatform sets the CPU Affinity mask of the program to run
// on the specified CPU cores based on the given platform string of
// rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588
func SetCPUAffinityByPlatform(platform string, ct CoreType) error {

	var useCores uintptr
	matched := false

	platform = strings.TrimSpace(platform)
	platform = strings.ToLower(platform)

	if platform, ok := coreMaskList[platform]; ok {
		if coretype, ok := platform[ct]; ok {
			useCores = coretype
			matched = true
		}
	}

	if !matched {
		return fmt.Errorf("unknown platform: %s", platform)
	}

	return SetCPUAffinity(useCores)
}
