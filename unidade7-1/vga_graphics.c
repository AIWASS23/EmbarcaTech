#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/dma.h"

#include "hsync.pio.h"
#include "vsync.pio.h"
#include "rgb.pio.h"
#include "vga_graphics.h"
#include "font.c"

#define H_ACTIVE   655    
#define V_ACTIVE   479    
#define RGB_ACTIVE 319    

#define TXCOUNT 153600 

unsigned char vga_data_array[TXCOUNT];
char * address_pointer = &vga_data_array[0] ;

#define TOPMASK 0b11000111
#define BOTTOMMASK 0b11111000

#define swap(a, b) { short t = a; a = b; b = t; }

#define tabspace 4 

#define pgm_read_byte(addr) (*(const unsigned char *)(addr))

unsigned short cursor_y, cursor_x, textsize ;
char textcolor, textbgcolor, wrap;

#define _width 640
#define _height 480

void initVGA() {
    PIO pio = pio0;

    uint hsync_offset = pio_add_program(pio, &hsync_program);
    uint vsync_offset = pio_add_program(pio, &vsync_program);
    uint rgb_offset = pio_add_program(pio, &rgb_program);

    uint hsync_sm = 0;
    uint vsync_sm = 1;
    uint rgb_sm = 2;

    hsync_program_init(pio, hsync_sm, hsync_offset, HSYNC);
    vsync_program_init(pio, vsync_sm, vsync_offset, VSYNC);
    rgb_program_init(pio, rgb_sm, rgb_offset, RED_PIN);

    int rgb_chan_0 = 0;
    int rgb_chan_1 = 1;

    dma_channel_config c0 = dma_channel_get_default_config(rgb_chan_0);  
    channel_config_set_transfer_data_size(&c0, DMA_SIZE_8);              
    channel_config_set_read_increment(&c0, true);                        
    channel_config_set_write_increment(&c0, false);                      
    channel_config_set_dreq(&c0, DREQ_PIO0_TX2) ;                       
    channel_config_set_chain_to(&c0, rgb_chan_1);                        

    dma_channel_configure(
        rgb_chan_0,                 
        &c0,                        
        &pio->txf[rgb_sm],          
        &vga_data_array,            
        TXCOUNT,                    
        false                       
    );

    dma_channel_config c1 = dma_channel_get_default_config(rgb_chan_1);   
    channel_config_set_transfer_data_size(&c1, DMA_SIZE_32);              
    channel_config_set_read_increment(&c1, false);                        
    channel_config_set_write_increment(&c1, false);                       
    channel_config_set_chain_to(&c1, rgb_chan_0);                         

    dma_channel_configure(
        rgb_chan_1,                         
        &c1,                                
        &dma_hw->ch[rgb_chan_0].read_addr,  
        &address_pointer,                   
        1,                                  
        false                               
    );

    pio_sm_put_blocking(pio, hsync_sm, H_ACTIVE);
    pio_sm_put_blocking(pio, vsync_sm, V_ACTIVE);
    pio_sm_put_blocking(pio, rgb_sm, RGB_ACTIVE);

    pio_enable_sm_mask_in_sync(pio, ((1u << hsync_sm) | (1u << vsync_sm) | (1u << rgb_sm)));

    dma_start_channel_mask((1u << rgb_chan_0)) ;
}

void drawPixel(short x, short y, char color) {
    if (x > 639) x = 639 ;
    if (x < 0) x = 0 ;
    if (y < 0) y = 0 ;
    if (y > 479) y = 479 ;

    int pixel = ((640 * y) + x) ;

    if (pixel & 1) {
      vga_data_array[pixel>>1] = (vga_data_array[pixel>>1] & TOPMASK) | (color << 3) ;
    } else {
      vga_data_array[pixel>>1] = (vga_data_array[pixel>>1] & BOTTOMMASK) | (color) ;
    }
}

void drawVLine(short x, short y, short h, char color) {
    for (short i=y; i<(y+h); i++) {
        drawPixel(x, i, color) ;
    }
}

void drawHLine(short x, short y, short w, char color) {
    for (short i=x; i<(x+w); i++) {
        drawPixel(i, y, color) ;
    }
}

void drawLine(short x0, short y0, short x1, short y1, char color) {
      short steep = abs(y1 - y0) > abs(x1 - x0);
      if (steep) {
        swap(x0, y0);
        swap(x1, y1);
      }

      if (x0 > x1) {
        swap(x0, x1);
        swap(y0, y1);
      }

      short dx, dy;
      dx = x1 - x0;
      dy = abs(y1 - y0);

      short err = dx / 2;
      short ystep;

      if (y0 < y1) {
        ystep = 1;
      } else {
        ystep = -1;
      }

      for (; x0<=x1; x0++) {
        if (steep) {
          drawPixel(y0, x0, color);
        } else {
          drawPixel(x0, y0, color);
        }
        err -= dy;
        if (err < 0) {
          y0 += ystep;
          err += dx;
        }
      }
}

void drawRect(short x, short y, short w, short h, char color) {

  drawHLine(x, y, w, color);
  drawHLine(x, y+h-1, w, color);
  drawVLine(x, y, h, color);
  drawVLine(x+w-1, y, h, color);
}

void drawCircle(short x0, short y0, short r, char color) {

  short f = 1 - r;
  short ddF_x = 1;
  short ddF_y = -2 * r;
  short x = 0;
  short y = r;

  drawPixel(x0  , y0+r, color);
  drawPixel(x0  , y0-r, color);
  drawPixel(x0+r, y0  , color);
  drawPixel(x0-r, y0  , color);

  while (x<y) {
    if (f >= 0) {
      y--;
      ddF_y += 2;
      f += ddF_y;
    }
    x++;
    ddF_x += 2;
    f += ddF_x;

    drawPixel(x0 + x, y0 + y, color);
    drawPixel(x0 - x, y0 + y, color);
    drawPixel(x0 + x, y0 - y, color);
    drawPixel(x0 - x, y0 - y, color);
    drawPixel(x0 + y, y0 + x, color);
    drawPixel(x0 - y, y0 + x, color);
    drawPixel(x0 + y, y0 - x, color);
    drawPixel(x0 - y, y0 - x, color);
  }
}

void drawCircleHelper( short x0, short y0, short r, unsigned char cornername, char color) {
  short f     = 1 - r;
  short ddF_x = 1;
  short ddF_y = -2 * r;
  short x     = 0;
  short y     = r;

  while (x<y) {
    if (f >= 0) {
      y--;
      ddF_y += 2;
      f     += ddF_y;
    }
    x++;
    ddF_x += 2;
    f     += ddF_x;
    if (cornername & 0x4) {
      drawPixel(x0 + x, y0 + y, color);
      drawPixel(x0 + y, y0 + x, color);
    }
    if (cornername & 0x2) {
      drawPixel(x0 + x, y0 - y, color);
      drawPixel(x0 + y, y0 - x, color);
    }
    if (cornername & 0x8) {
      drawPixel(x0 - y, y0 + x, color);
      drawPixel(x0 - x, y0 + y, color);
    }
    if (cornername & 0x1) {
      drawPixel(x0 - y, y0 - x, color);
      drawPixel(x0 - x, y0 - y, color);
    }
  }
}

void fillCircle(short x0, short y0, short r, char color) {
  drawVLine(x0, y0-r, 2*r+1, color);
  fillCircleHelper(x0, y0, r, 3, 0, color);
}

void fillCircleHelper(short x0, short y0, short r, unsigned char cornername, short delta, char color) {
  short f     = 1 - r;
  short ddF_x = 1;
  short ddF_y = -2 * r;
  short x     = 0;
  short y     = r;

  while (x<y) {
    if (f >= 0) {
      y--;
      ddF_y += 2;
      f     += ddF_y;
    }
    x++;
    ddF_x += 2;
    f     += ddF_x;

    if (cornername & 0x1) {
      drawVLine(x0+x, y0-y, 2*y+1+delta, color);
      drawVLine(x0+y, y0-x, 2*x+1+delta, color);
    }
    if (cornername & 0x2) {
      drawVLine(x0-x, y0-y, 2*y+1+delta, color);
      drawVLine(x0-y, y0-x, 2*x+1+delta, color);
    }
  }
}

void drawRoundRect(short x, short y, short w, short h, short r, char color) {

  drawHLine(x+r  , y    , w-2*r, color); // Top
  drawHLine(x+r  , y+h-1, w-2*r, color); // Bottom
  drawVLine(x    , y+r  , h-2*r, color); // Left
  drawVLine(x+w-1, y+r  , h-2*r, color); // Right

  drawCircleHelper(x+r    , y+r    , r, 1, color);
  drawCircleHelper(x+w-r-1, y+r    , r, 2, color);
  drawCircleHelper(x+w-r-1, y+h-r-1, r, 4, color);
  drawCircleHelper(x+r    , y+h-r-1, r, 8, color);
}

void fillRoundRect(short x, short y, short w, short h, short r, char color) {
  fillRect(x+r, y, w-2*r, h, color);

  fillCircleHelper(x+w-r-1, y+r, r, 1, h-2*r-1, color);
  fillCircleHelper(x+r    , y+r, r, 2, h-2*r-1, color);
}

void fillRect(short x, short y, short w, short h, char color) {

  for(int i=x; i<(x+w); i++) {
    for(int j=y; j<(y+h); j++) {
        drawPixel(i, j, color);
    }
  }
}

void drawChar(short x, short y, unsigned char c, char color, char bg, unsigned char size) {
    char i, j;
  if((x >= _width)            || // Clip right
     (y >= _height)           || // Clip bottom
     ((x + 6 * size - 1) < 0) || // Clip left
     ((y + 8 * size - 1) < 0))   // Clip top
    return;

  for (i=0; i<6; i++ ) {
    unsigned char line;
    if (i == 5)
      line = 0x0;
    else
      line = pgm_read_byte(font+(c*5)+i);
    for ( j = 0; j<8; j++) {
      if (line & 0x1) {
        if (size == 1) 
          drawPixel(x+i, y+j, color);
        else {  
          fillRect(x+(i*size), y+(j*size), size, size, color);
        }
      } else if (bg != color) {
        if (size == 1) 
          drawPixel(x+i, y+j, bg);
        else {  
          fillRect(x+i*size, y+j*size, size, size, bg);
        }
      }
      line >>= 1;
    }
  }
}


inline void setCursor(short x, short y) {
  cursor_x = x;
  cursor_y = y;
}

inline void setTextSize(unsigned char s) {
  textsize = (s > 0) ? s : 1;
}

inline void setTextColor(char c) {
  textcolor = textbgcolor = c;
}

inline void setTextColor2(char c, char b) {
  textcolor   = c;
  textbgcolor = b;
}

inline void setTextWrap(char w) {
  wrap = w;
}


void tft_write(unsigned char c){
  if (c == '\n') {
    cursor_y += textsize*8;
    cursor_x  = 0;
  } else if (c == '\r') {
  } else if (c == '\t'){
      int new_x = cursor_x + tabspace;
      if (new_x < _width){
          cursor_x = new_x;
      }
  } else {
    drawChar(cursor_x, cursor_y, c, textcolor, textbgcolor, textsize);
    cursor_x += textsize*6;
    if (wrap && (cursor_x > (_width - textsize*6))) {
      cursor_y += textsize*8;
      cursor_x = 0;
    }
  }
}

inline void writeString(char* str){
    while (*str){
        tft_write(*str++);
    }
}