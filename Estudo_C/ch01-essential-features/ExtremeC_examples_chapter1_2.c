// File name: ExtremeC_exampels_chapter1_2.c
// Description: Example 1.2

#include<stdio.h>
#define ADD(a, b) a + b


int main(int argc, char** argv) {
  int x = 2;
  int y = 3;
  int z = ADD(x, y);
  printf("%d", z);
  return 0;
}
