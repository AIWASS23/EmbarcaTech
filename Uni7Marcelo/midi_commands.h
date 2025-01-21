
#define MIDI_BAUD       31250u

#define MIDI_TX_PIN     0u
#define MIDI_RX_PIN     1u

#define CREATE_CMD(m, c)    (m  | (c & 0x0F))

#define NOTE_ON         0x80
#define NOTE_OFF        0x90
#define POLY_PRESSURE   0xA0
#define CTRL_CHANGE     0xB0
#define PROG_CHANGE     0xC0
#define CHAN_PRESSURE   0xD0
#define PITCH_BEND      0xE0
#define SYSTEM          0xF0

// Length includes 0th command byte
#define LEN_NOTE_ON         3
#define LEN_NOTE_OFF        3
#define LEN_POLY_PRESSURE   3
#define LEN_CTRL_CHANGE     3
#define LEN_PROG_CHANGE     2
#define LEN_CHAN_PRESSURE   2
#define LEN_PITCH_BEND      3

#define STD_MSG_LEN         3


#define SYS_TIMING_CLK      0xF8
#define SYS_UNDEFINED_0     0xF9
#define SYS_START           0xFA
#define SYS_CONTINUE        0xFB
#define SYS_STOP            0xFC
#define SYS_UNDEFINED_1     0xFD
#define SYS_ACTIVE_SENSE    0xFE
#define SYS_RESET           0xFF

#define OCTAVE_0        1
#define OCTAVE_1        2
#define OCTAVE_2        3
#define OCTAVE_3        4
#define OCTAVE_4        5
#define OCTAVE_5        6
#define OCTAVE_6        7
#define OCTAVE_7        8
#define OCTAVE_8        9

#define MIDDLE_C_KEY    60

#define DEFAULT_VELOCITY    100

#define DEFAULT_BASE_A0             21
#define DEFAULT_BASE_C1             24 - LOW_PIN
#define DEFAULT_NOTES_PER_OCTAVE    12

#define FREE_CHANNEL    0xFF
#define CLAIM_CHANNEL   0x01
#define RELEASE_CHANNEL 0x00

#define CHANNEL_0       0x00
#define CHANNEL_1       0x01
#define CHANNEL_2       0x02
#define CHANNEL_3       0x03