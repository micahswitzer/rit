/*
 * This is a toy linker script for building programs
 * that have a simple, but "realistic" layout
 */

OUTPUT_ARCH("riscv")

SECTIONS
{
    .text 0x1000 : {
        _stack = .;
        *(.text*)
    }
    .data 0x2000 : {
        *(.data*)
        *(.rodata*)
    }
}
