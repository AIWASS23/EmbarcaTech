#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/i2c.h"
#include "hardware/timer.h"
#include "hardware/pwm.h"

#include "midi_commands.h"
#include "bsp/board.h"
#include "tusb.h"

#define LED_PIN         25ul    
#define FAST_I2C_FREQ   400000  
#define I2C_ADDR        0x0A    
#define I2C_SCL         3
#define I2C_SDA         2

#define NOTE_C_LOWER    15ul    // GPIO 15  Lower C     // Key 0
#define NOTE_D_FLAT     14ul    // GPIO 14  D flat / C# // Key 1
#define NOTE_D          13ul    // GPIO 13  D           // Key 2
#define NOTE_E_FLAT     12ul    // GPIO 12  E flat / D# // Key 3
#define NOTE_E          11ul    // GPIO 11  E           // Key 4
#define NOTE_F          10ul    // GPIO 10  F           // Key 5

#define NOTE_G_FLAT     16ul    // GPIO 16  G flat / F# // Key 6
#define NOTE_G          17ul    // GPIO 17  G           // Key 7
#define NOTE_A_FLAT     18ul    // GPIO 18  A flat / G# // Key 8
#define NOTE_A          19ul    // GPIO 19  A           // Key 9
#define NOTE_B_FLAT     20ul    // GPIO 20  B flat / A# // Key 10
#define NOTE_B          21ul    // GPIO 21  B           // Key 11
#define NOTE_C_UPPER    22ul    // GPIO 22  Upper C     // Key 12

#define NUM_KEYS        13
#define NUM_VOICES      4
#define LOW_PIN         10ul
#define HIGH_PIN        22ul


#define KEYBOARD_MASK   (   (1u << NOTE_C_LOWER) |\
                            (1u << NOTE_D_FLAT) |\
                            (1u << NOTE_D) |\
                            (1u << NOTE_E_FLAT) |\
                            (1u << NOTE_E) |\
                            (1u << NOTE_F) |\
                            (1u << NOTE_G_FLAT) |\
                            (1u << NOTE_G) |\
                            (1u << NOTE_A_FLAT) |\
                            (1u << NOTE_A) |\
                            (1u << NOTE_B_FLAT) |\
                            (1u << NOTE_B) |\
                            (1u << NOTE_C_UPPER) \
                        )

#define ENCODER_BUTTON  9ul     // GPIO 9   Encoder Push Button

#define CHANGES_IN_VAL(a, b)    (a ^ b)
#define DID_PIN_CHANGE(a, b, n) ( (a & (1 << n) ^ (b & (1 << n)) ) ? 0 : 1)

struct KeyboardKeys {
    uint32_t key_mask;
    uint32_t last_key_read;
    uint8_t key_gpio[NUM_KEYS];
    uint8_t key_note[NUM_KEYS];
    uint8_t current_octave;
    uint8_t key_velocity[NUM_KEYS];
};


struct VoiceStates{
    uint8_t voices_free;
    uint8_t voice_pin[NUM_VOICES]; 
};


struct MIDI_MESSAGE{
    uint8_t length;
    uint8_t payload[3];
};

void core_1_entry();      
void setup_i2C(i2c_inst_t* i2c);

void keyboard_struct_init(struct KeyboardKeys *keystruct);
void setup_Keyboard_Keys(struct KeyboardKeys *keystruct);

uint init_midi(uart_inst_t *uart);
void send_midi_message(uart_inst_t *uart, struct MIDI_MESSAGE *msg);

uint32_t process_keyboard_inputs(uart_inst_t *uart, struct KeyboardKeys *keyboard, struct VoiceStates *voiceStates);
int8_t channel_arbitration(struct VoiceStates *voices, uint8_t pin_requesting, uint8_t operation);


void check_for_encoder_press(struct KeyboardKeys *keyboard);
void midi_task(uart_inst_t *uart);

int main(){                
    board_init();
    tusb_init();

    multicore_launch_core1(core_1_entry);

   struct VoiceStates voices;
   voices.voices_free = NUM_VOICES;
   for(uint n = 0; n < NUM_VOICES; n++) voices.voice_pin[n] = FREE_CHANNEL;

    struct KeyboardKeys keyboard;
    keyboard.key_mask = KEYBOARD_MASK;
    

    keyboard_struct_init(&keyboard);
    setup_Keyboard_Keys(&keyboard);

    init_midi(uart0);

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 1);

    busy_wait_ms(500);

    gpio_put(LED_PIN, 0);

    while(1){
        if(gpio_get(ENCODER_BUTTON)) process_keyboard_inputs(uart0, &keyboard, &voices);
        else check_for_encoder_press(&keyboard);


        busy_wait_ms(50);
    }

}

void core_1_entry(){        
    while(1){
        tud_task(); 
        midi_task(uart0);
    } 

}


void midi_task(uart_inst_t *uart){

    uint8_t packet[4];
    while ( tud_midi_available() ) tud_midi_packet_read(packet);

    if((packet[0] & 0xF0 == NOTE_ON) || (packet[0] & 0xF0 == NOTE_OFF)){
        gpio_put(LED_PIN, ~gpio_get(LED_PIN));
        struct MIDI_MESSAGE msg;
        msg.length = STD_MSG_LEN;
        msg.payload[0] = packet[0];
        msg.payload[1] = packet[1];
        msg.payload[2] = packet[2];        
    } else
    if((packet[1] & 0xF0 == NOTE_ON) || (packet[1] & 0xF0 == NOTE_OFF)){
        gpio_put(LED_PIN, ~gpio_get(LED_PIN));
        struct MIDI_MESSAGE msg;
        msg.length = STD_MSG_LEN;
        msg.payload[0] = packet[1];
        msg.payload[1] = packet[2];
        msg.payload[2] = packet[3];        
    }
}


void setup_i2C(i2c_inst_t* i2c){
    gpio_set_function(I2C_SDA, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SCL);
    gpio_pull_up(I2C_SDA);
    i2c_init(i2c, FAST_I2C_FREQ);
    i2c_set_slave_mode(i2c, 0, I2C_ADDR);
}

void setup_Keyboard_Keys(struct KeyboardKeys *keystruct){

    gpio_init(ENCODER_BUTTON);
    gpio_set_pulls(ENCODER_BUTTON, 1, 0);
    gpio_set_input_enabled(ENCODER_BUTTON, 1);

    gpio_init_mask(keystruct->key_mask);
    gpio_set_dir_in_masked(keystruct->key_mask);
    
    keystruct->current_octave = OCTAVE_4;

    for(uint32_t n = 0; n < NUM_KEYS; n++){
        keystruct->key_velocity[n] = DEFAULT_VELOCITY;
    }

    for(uint32_t n = 0; n < NUM_BANK0_GPIOS; n++){
        if(keystruct->key_mask & (1 << n)) gpio_set_pulls(n, 1, 0);
    }
}

void keyboard_struct_init(struct KeyboardKeys *keystruct){
    keystruct->key_mask = KEYBOARD_MASK;
    keystruct->last_key_read = 0;
                                                // GPIO #
    keystruct->key_gpio[0]  = NOTE_F;           // 10
    keystruct->key_gpio[1]  = NOTE_E;           // 11
    keystruct->key_gpio[2]  = NOTE_E_FLAT;      // 12
    keystruct->key_gpio[3]  = NOTE_D;           // 13

    keystruct->key_gpio[4]  = NOTE_D_FLAT;      // 14
    keystruct->key_gpio[5]  = NOTE_C_LOWER;     // 15
    keystruct->key_gpio[6]  = NOTE_G_FLAT;      // 16
    keystruct->key_gpio[7]  = NOTE_G;           //17
    keystruct->key_gpio[8]  = NOTE_A_FLAT;      // 18

    keystruct->key_gpio[9]  = NOTE_A;           // 19
    keystruct->key_gpio[10] = NOTE_B_FLAT;      // 20
    keystruct->key_gpio[11] = NOTE_B;           // 21
    keystruct->key_gpio[12] = NOTE_C_UPPER;     // 22

    // Note offsets
    keystruct->key_note[0] = 5;                 // 10
    keystruct->key_note[1] = 4;                 // 11
    keystruct->key_note[2] = 3;                 // 12
    keystruct->key_note[3] = 2;                 // 13

    keystruct->key_note[4] = 1;                 // 14
    keystruct->key_note[5] = 0;                 // 15
    keystruct->key_note[6] = 6;                 // 16
    keystruct->key_note[7] = 7;                 // 17
    keystruct->key_note[8] = 8;                 // 18

    keystruct->key_note[9] = 9;                 // 19
    keystruct->key_note[10] = 10;               // 20
    keystruct->key_note[11] = 11;               // 21
    keystruct->key_note[12] = 12;               // 22
}

uint init_midi(uart_inst_t *uart){
    uint actual_baud = uart_init(uart, MIDI_BAUD);
    uart_set_format(uart, 8, 1, UART_PARITY_NONE);
    uart_set_fifo_enabled(uart, 1);

    gpio_set_function(MIDI_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(MIDI_RX_PIN, GPIO_FUNC_UART);

    return actual_baud;
}


void send_midi_message(uart_inst_t *uart, struct MIDI_MESSAGE *msg){
    for(uint n = 0; n < msg->length; n++){
        uart_putc_raw(uart, msg->payload[n]);
    }

}


int8_t channel_arbitration(struct VoiceStates *voices, uint8_t pin_requesting, uint8_t operation){
    int8_t ret_channel = -1;


    if(operation){                              
        for(int8_t n = 0; n < NUM_VOICES; n++){
            if((voices->voice_pin[n] == pin_requesting) || (voices->voice_pin[n] == FREE_CHANNEL)){
                voices->voice_pin[n] = pin_requesting;
                ret_channel = n;
                break;
            }
        }
    } else {                                    
        for(int8_t n = 0; n < NUM_VOICES; n++){
            if((voices->voice_pin[n] == pin_requesting)){
                voices->voice_pin[n] = FREE_CHANNEL;
                ret_channel = n;
                break;
            }
        }
    }
    
    return ret_channel;
}


uint32_t process_keyboard_inputs(uart_inst_t *uart, struct KeyboardKeys *keyboard, struct VoiceStates *voiceStates){
    uint32_t gpio_vals = (gpio_get_all());
    uint32_t retval = 0;

    uint32_t changed_pins = (keyboard->key_mask & CHANGES_IN_VAL(keyboard->last_key_read, gpio_vals));
    if(changed_pins){
        for(uint32_t n = LOW_PIN; n < (HIGH_PIN + 1); n++){
            if(changed_pins & (1 << n)){
                struct MIDI_MESSAGE msg;
                msg.length = STD_MSG_LEN;
                
                int8_t ch_arb = channel_arbitration(voiceStates, (uint8_t)n, ((gpio_vals & (changed_pins & (1 << n))) ? 0 : 1));
                if(ch_arb > -1){
                    msg.payload[0] = (( gpio_get(n) != 0 ) ? NOTE_ON : NOTE_OFF) | ch_arb;
                    msg.payload[1] = (keyboard->key_note[n - LOW_PIN]) + (keyboard->current_octave * 0x0C);     // key pressed is an offset from base C note (set by octave selection)
                    msg.payload[2] = keyboard->key_velocity[(n - LOW_PIN)];                                     // If velocity is enabled, apply here
                    send_midi_message(uart0, &msg);
                }
            }
        }

        keyboard->last_key_read = gpio_vals;
    }

    return retval;
}

void check_for_encoder_press(struct KeyboardKeys *keyboard){
    static uint8_t current_wave = 0;

    if(!gpio_get(ENCODER_BUTTON)){                  
             
            gpio_put(LED_PIN, 1);
            busy_wait_ms(300);
            gpio_put(LED_PIN, 0);

            switch(current_wave){
                case 0:                    
                    busy_wait_ms(500);
                    gpio_put(LED_PIN, 1);
                break;

                case 1:                    
                    busy_wait_ms(250);
                    gpio_put(LED_PIN, 1);
                    busy_wait_ms(250);
                    gpio_put(LED_PIN, 0);
                break;

                case 2:                     
                    busy_wait_ms(150);
                    gpio_put(LED_PIN, 1);
                    busy_wait_ms(150);
                    gpio_put(LED_PIN, 0);
                    busy_wait_ms(150);
                    gpio_put(LED_PIN, 1);
                    busy_wait_ms(150);
                    gpio_put(LED_PIN, 0);
                break;

                case 3:                     
                    busy_wait_ms(100);
                    gpio_put(LED_PIN, 1);
                    busy_wait_ms(100);
                    gpio_put(LED_PIN, 0);
                    busy_wait_ms(100);
                    gpio_put(LED_PIN, 1);
                    busy_wait_ms(100);
                    gpio_put(LED_PIN, 0);
                    busy_wait_ms(100);
                    gpio_put(LED_PIN, 1);
                    busy_wait_ms(100);
                    gpio_put(LED_PIN, 0);
                break;
            }
            
            busy_wait_ms(200);
            gpio_put(LED_PIN, 1);

            struct MIDI_MESSAGE wave_change;
            wave_change.length = STD_MSG_LEN;

            wave_change.payload[1] = 0x00;      
            wave_change.payload[2] = current_wave;

            while(!gpio_get(ENCODER_BUTTON)){
                if(!gpio_get(NOTE_C_LOWER)){
                    wave_change.payload[0] = CREATE_CMD(CTRL_CHANGE, CHANNEL_0);
                    send_midi_message(uart0, &wave_change);
                    busy_wait_ms(100);    
                } else
                if(!gpio_get(NOTE_D)){
                    wave_change.payload[0] = CREATE_CMD(CTRL_CHANGE, CHANNEL_1);
                    send_midi_message(uart0, &wave_change);
                    busy_wait_ms(100);
                } else
                if(!gpio_get(NOTE_E)){
                    wave_change.payload[0] = CREATE_CMD(CTRL_CHANGE, CHANNEL_2);
                    send_midi_message(uart0, &wave_change);
                    busy_wait_ms(100);
                } else
                if(!gpio_get(NOTE_F)){
                    wave_change.payload[0] = CREATE_CMD(CTRL_CHANGE, CHANNEL_3);
                    send_midi_message(uart0, &wave_change);
                    busy_wait_ms(100);
                } else

                if(!gpio_get(NOTE_C_UPPER)){        
                    if(keyboard->current_octave < OCTAVE_8) keyboard->current_octave += 1;
                    busy_wait_ms(400);
                } else
                if(!gpio_get(NOTE_B)){              
                    if(keyboard->current_octave > OCTAVE_1) keyboard->current_octave -= 1;
                    busy_wait_ms(400);
                }
            }
            

            if(current_wave < 3) current_wave += 1;
            else current_wave = 0;

            busy_wait_ms(10);
        
        }
        
        if(gpio_get(LED_PIN)) gpio_put(LED_PIN, 0);
}