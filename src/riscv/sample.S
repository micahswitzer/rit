.option relax

.section .rodata
.local _message
_message: .ascii "hello world!\n"
_message_end: .byte 0
#ifdef __clang__
    # LLVM is quite strange and for some reason refuses to assemble
    # a `li` with any value that is not a plain int, even if it's known
    # at assemble-time...
    .equ _message_size, 13
#else
    # the GNU assembler handles this just fine...
    .equ _message_size, _message_end - _message
#endif

.extern _stack, _main

.section .text
.global _start
_start:
    ## set up the stack pointer
    la     sp, _stack
    ## clear the return address
    li     ra, 0
    ## load arguments
    # a0: pointer to byte array
    # a1: size of byte array
    la     a0, _message
    li     a1, _message_size
    ## call into our C code
    call   _main
    # TODO: use the returned value?
.halt:
    ebreak
    j .halt
