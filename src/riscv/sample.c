/*
 * this file exports a few simple functions that are designed to generate simple, but non-trival assembly
 */

unsigned int checksum(const void *start, const void *end)
{
    const unsigned char *buf = (unsigned char *)start;
    unsigned char sum = 0;
    do {
        // integer overflow is undefined behavior in C
        // so be specific about the behavior we want
        sum = (unsigned char)((unsigned int)sum + (unsigned int)*buf);
    } while (++buf != end);
    return sum;
}

unsigned int _main(const void *data, unsigned int size)
{
    return checksum(data, data + size);
}
