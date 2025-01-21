#ifndef __PT_H__
#define __PT_H__

#ifdef DOXYGEN

#define LC_INIT(lc)

#define LC_SET(lc)

#define LC_RESUME(lc)

#define LC_END(lc)

#endif 

#ifndef __LC_ADDRLABELS_H__
#define __LC_ADDRLABELS_H__

typedef void * lc_t;

#define LC_INIT(s) s = NULL

#define LC_RESUME(s) \
  do { \
    if(s != NULL) { goto *s; } \
  } while(0)

#define LC_CONCAT2(s1, s2) s1##s2
#define LC_CONCAT(s1, s2) LC_CONCAT2(s1, s2)

#define LC_SET(s)  do { LC_CONCAT(LC_LABEL, __LINE__): (s) = &&LC_CONCAT(LC_LABEL, __LINE__); } while(0)
#define LC_END(s)

#endif 

struct pt {
  lc_t lc;
};

#define PT_WAITING 0
#define PT_YIELDED 1
#define PT_EXITED  2
#define PT_ENDED   3

#define PT_INIT(pt)   LC_INIT((pt)->lc)

#define PT_THREAD(name_args) char name_args

#define PT_BEGIN(pt) { char PT_YIELD_FLAG = 1; LC_RESUME((pt)->lc)

#define PT_END(pt) LC_END((pt)->lc); PT_YIELD_FLAG = 0; PT_INIT(pt); return PT_ENDED; }

#define PT_WAIT_UNTIL(pt, condition) do { LC_SET((pt)->lc); if(!(condition)) { return PT_WAITING; } } while(0)

#define PT_WAIT_WHILE(pt, cond)  PT_WAIT_UNTIL((pt), !(cond))

#define PT_WAIT_THREAD(pt, thread) PT_WAIT_WHILE((pt), PT_SCHEDULE(thread))

#define PT_SPAWN(pt, child, thread) do { PT_INIT((child)); PT_WAIT_THREAD((pt), (thread)); } while(0)

#define PT_RESTART(pt) do { PT_INIT(pt); return PT_WAITING; } while(0)

#define PT_EXIT(pt) do { PT_INIT(pt); return PT_EXITED; } while(0)

#define PT_SCHEDULE(f) ((f) < PT_EXITED)

#define PT_YIELD(pt) do { PT_YIELD_FLAG = 0; LC_SET((pt)->lc); if(PT_YIELD_FLAG == 0) { return PT_YIELDED; } } while(0)

#define PT_YIELD_UNTIL(pt, cond) do { PT_YIELD_FLAG = 0; LC_SET((pt)->lc); if((PT_YIELD_FLAG == 0) || !(cond)) { return PT_YIELDED; } } while(0)

#endif 

#ifndef __PT_SEM_H__
#define __PT_SEM_H__

struct pt_sem {
  unsigned int count;
};

#define PT_SEM_INIT(s, c) (s)->count = c

#define PT_SEM_WAIT(pt, s) do { PT_YIELD_UNTIL(pt, (s)->count > 0); --(s)->count; } while(0)

#define PT_SEM_SIGNAL(pt, s) ++(s)->count

#endif 

#define PT_YIELD_usec(delay_time) do { static unsigned int time_thread ; time_thread = timer_hw->timerawl + (unsigned int)delay_time ; PT_YIELD_UNTIL(pt, (timer_hw->timerawl >= time_thread)); } while(0);

#define PT_GET_TIME_usec() (timer_hw->timerawl)

#define PT_INTERVAL_INIT() static unsigned int pt_interval_marker

#define PT_YIELD_INTERVAL(interval_time) do { PT_YIELD_UNTIL(pt, (timer_hw->timerawl >= pt_interval_marker)); pt_interval_marker = timer_hw->timerawl + (unsigned int)interval_time; } while(0);

spin_lock_t * sem_lock ;

#define PT_SEM_SAFE_INIT(s,c) do{ sem_lock = spin_lock_init(25); spin_lock_unsafe_blocking (sem_lock); (s)->count = c ; spin_unlock_unsafe (sem_lock); } while(0)

#define PT_SEM_SAFE_WAIT(pt,s) do { spin_lock_unsafe_blocking (sem_lock); PT_YIELD_FLAG = 0; LC_SET((pt)->lc); if((PT_YIELD_FLAG == 0) || !((s)->count > 0)) { spin_unlock_unsafe (sem_lock); return PT_YIELDED; } --(s)->count; spin_unlock_unsafe (sem_lock); } while(0)

#define PT_SEM_SAFE_SIGNAL(pt,s) do{ spin_lock_unsafe_blocking (sem_lock); ++(s)->count ; spin_unlock_unsafe (sem_lock) ;} while(0)

#define UNLOCKED 0
#define LOCKED 1
spin_lock_t * lock_lock ;

#define PT_LOCK_INIT(s,lock_num,lock_state) do{ \
  lock_lock = spin_lock_init(24); \
  spin_lock_unsafe_blocking (lock_lock); \
  s = spin_lock_init((uint)lock_num); \
  if(lock_state) spin_lock_unsafe_blocking (s); \
  spin_unlock_unsafe (lock_lock) ; \
} while(0)

#define PT_LOCK_WAIT(pt,s)  do {  \
  spin_lock_unsafe_blocking (lock_lock); \
  PT_YIELD_FLAG = 0;        \
  LC_SET((pt)->lc);       \
  if((PT_YIELD_FLAG == 0) || !(is_spin_locked(s)==false)) { \
      spin_unlock_unsafe (lock_lock) ; \
      return PT_YIELDED;                        \
  }           \
  spin_lock_unsafe_blocking (s); \
  spin_unlock_unsafe (lock_lock) ; \
} while(0)

#define PT_LOCK_RELEASE(s) do{ \
    spin_unlock_unsafe (s) ; \
} while(0)

#define PT_FIFO_WRITE(data) do{ \
    PT_YIELD_UNTIL(pt, multicore_fifo_wready()==true); \
    multicore_fifo_push_blocking(data) ; \
} while(0)

#define PT_FIFO_READ(fifo_out)  \
do{ \
    PT_YIELD_UNTIL(pt, multicore_fifo_rvalid()==true); \
    fifo_out = multicore_fifo_pop_blocking() ; \
} while(0) 


#define PT_FIFO_FLUSH do{ \
    multicore_fifo_drain() ; \
} while(0)

static struct pt pt_sched ;
static struct pt pt_sched1 ;

int pt_task_count = 0 ;
int pt_task_count1 = 0 ;

struct ptx {
  struct pt pt;              
  int num;                    
  char (*pf)(struct pt *pt); 
};

#define MAX_THREADS 10
static struct ptx pt_thread_list[MAX_THREADS];
static struct ptx pt_thread_list1[MAX_THREADS];

int pt_add( char (*pf)(struct pt *pt)) {
  if (pt_task_count < (MAX_THREADS)) {
    struct ptx *ptx = &pt_thread_list[pt_task_count];
    ptx->num   = pt_task_count;
    ptx->pf    = pf;
    PT_INIT( &ptx->pt );
    pt_task_count++;
        return pt_task_count-1;
  }
  return 0;
}

int pt_add1( char (*pf)(struct pt *pt)) {
  if (pt_task_count1 < (MAX_THREADS)) {
    struct ptx *ptx = &pt_thread_list1[pt_task_count1];
    ptx->num   = pt_task_count1;
    ptx->pf    = pf;
    PT_INIT( &ptx->pt );
    pt_task_count1++;
        return pt_task_count1-1;
  }
  return 0;
}


#define SCHED_ROUND_ROBIN 0
#define SCHED_RATE 1
int pt_sched_method = SCHED_ROUND_ROBIN ;

static PT_THREAD (protothread_sched(struct pt *pt)) {   
    PT_BEGIN(pt);
    static int i, rate;
    
    if (pt_sched_method==SCHED_ROUND_ROBIN){
        while(1) {
          struct ptx *ptx = &pt_thread_list[0];
          for (i=0; i<pt_task_count; i++, ptx++ ){
              (pt_thread_list[i].pf)(&ptx->pt); 
          }
        } 
    }       
     
    PT_END(pt);
}

static PT_THREAD (protothread_sched1(struct pt *pt)) {   
    PT_BEGIN(pt);
    
    static int i, rate;
    
    if (pt_sched_method==SCHED_ROUND_ROBIN){
        while(1) {
          struct ptx *ptx = &pt_thread_list1[0];
          for (i=0; i<pt_task_count1; i++, ptx++ ){
              (pt_thread_list1[i].pf)(&ptx->pt); 
          }
        } 
    }       
     
    PT_END(pt);
}

#define pt_schedule_start do{\
  if(get_core_num()==1){ \
    PT_INIT(&pt_sched1) ; \
    PT_SCHEDULE(protothread_sched1(&pt_sched1));\
  }  else {\
    PT_INIT(&pt_sched) ;\
    PT_SCHEDULE(protothread_sched(&pt_sched));\
  }\
} while(0) 

#define pt_add_thread(thread_name) do{\
  if(get_core_num()==1){ \
    pt_add1(thread_name);\
  }  else {\
    pt_add(thread_name);\
  }\
} while(0) 


#define pt_buffer_size 100
char pt_serial_in_buffer[pt_buffer_size];
char pt_serial_out_buffer[pt_buffer_size];
static struct pt pt_serialin, pt_serialout ;

#define UART_ID uart0
#define pt_backspace 0x7f 

static PT_THREAD (pt_serialin_polled(struct pt *pt)){
    PT_BEGIN(pt);
      static uint8_t ch ;
      static int pt_current_char_count ;
      memset(pt_serial_in_buffer, 0, pt_buffer_size);
      pt_current_char_count = 0 ;
      while(uart_is_readable(UART_ID)){uart_getc(UART_ID);}
      while(pt_current_char_count < pt_buffer_size) {   
        PT_YIELD_UNTIL(pt, (int)uart_is_readable(UART_ID)) ;
        ch = uart_getc(UART_ID);
        PT_YIELD_UNTIL(pt, (int)uart_is_writable(UART_ID)) ;
        uart_putc(UART_ID, ch);
        if (ch == '\r' ){
          pt_serial_in_buffer[pt_current_char_count] = 0 ;
          PT_YIELD_UNTIL(pt, (int)uart_is_writable(UART_ID)) ;
          uart_putc(UART_ID, '\n') ;
          break ; 
        }

        else if (ch == pt_backspace){
          PT_YIELD_UNTIL(pt, (int)uart_is_writable(UART_ID)) ;
          uart_putc(UART_ID, ' ') ;
          PT_YIELD_UNTIL(pt, (int)uart_is_writable(UART_ID)) ;
          uart_putc(UART_ID, pt_backspace) ;
          
          pt_current_char_count-- ;
          if (pt_current_char_count<0) {pt_current_char_count = 0 ;}
        }
        else {
          pt_serial_in_buffer[pt_current_char_count++] = ch ;
        }
      } 

    PT_EXIT(pt);
  PT_END(pt);
} 

int pt_serialout_polled(struct pt *pt) {
    static int num_send_chars ;
    PT_BEGIN(pt);
    num_send_chars = 0;
    while (pt_serial_out_buffer[num_send_chars] != 0){
        PT_YIELD_UNTIL(pt, (int)uart_is_writable(UART_ID)) ;
        uart_putc(UART_ID, pt_serial_out_buffer[num_send_chars]) ;
        num_send_chars++;
    }
    PT_EXIT(pt);
    PT_END(pt);
}
#define serial_write do{PT_SPAWN(pt,&pt_serialout,pt_serialout_polled(&pt_serialout));}while(0)
#define serial_read  do{PT_SPAWN(pt,&pt_serialin,pt_serialin_polled(&pt_serialin));}while(0)