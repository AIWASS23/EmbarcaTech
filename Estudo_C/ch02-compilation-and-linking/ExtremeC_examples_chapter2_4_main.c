// File name: ExtremeC_examples_chapter2_4_main.c
// Description: This file contains the 'main' function.

#include "ExtremeC_examples_chapter2_4_decls.h"
#include<stdio.h>

int main(int argc, char** argv) {
  int x = add(4, 5);
  printf("%d\n", x);
  int y = multiply(9, x);
  printf("%d\n", y);
  return 0;
}
