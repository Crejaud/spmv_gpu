# cs516_spmv

Hi. the instructions to run this are very simple.

Simple type: make

then run ./spmv as normal

For the atomic and segment implementations, I recommend 64 blocks of size 1024.

For the design implementation, simply exchange the two, so 1024 blocks of size 32 or 64.
Sometimes the number of blocks you choose and the size of the block will cause an error, so choose wisely.

Any questions, please email me at crejaud@gmail.com

Thanks,
Corentin (Corey) Rejaud
