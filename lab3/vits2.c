/*********************************************************************/
/* vits:     Iterative Threshold Selection Algorithm               */
/*@ Copyright                                                      */
/*  Name: Yanling Wu                                               */
/*  NetID: yw996                                                   */
/*  ECE 5470 Lab3                                                  */
/*********************************************************************/

#include "VisXV4.h"          /* VisionX structure include file       */
#include "Vutil.h"           /* VisionX utility header files         */

VXparam_t par[] =            /* command line structure               */
{
{  "if=",   0,   " input file, vtpeak: threshold between hgram peaks"},
{  "of=",   0,   " output file "},
{  "t=",    0,   " the initial threshold (default the average of all pixel values)"},
{  "-v",    0,   "(verbose) print threshold information"},
{   0,      0,   0} /* list termination */
};
#define  IVAL   par[0].val
#define  OVAL   par[1].val
#define  TVAL   par[2].val
#define  VFLAG  par[3].val

main(argc, argv)
int argc;
char *argv[];
{

    Vfstruct (im);                 /* input image structure          */
    int y,x;                       /* index counters                 */
    int i;
    long sum1 = 0; 
    long sum2 = 0; 
    int count1 = 0;  
    int count2 = 0;
    int hist[256];                 /*histogram bins                  */
    int thresh;                    /* threshold                      */
    int orithresh = 127;                /* the original threshold   */
    long sum = 0;                /*the sum of the all pixel values    */
    int avg1 = 0;                /*the average of R1 (pixel above threshold) */
    int avg2 = 0;                /*the average of R2 (pixel below threshold) */
    int lasavg1 = 0;               
    int lasavg2 = 0;   
    int temp;  
    int count = 0;                     
                 
    VXparse(&argc, &argv, par);    /* parse the command line         */

    if (TVAL) orithresh = atoi(TVAL);  /* if d= was specified, get value */
    if (orithresh < 0 || orithresh > 255) {
    fprintf(stderr, "d= must be between 0 and 255\nUsing d=10\n");
        orithresh = 143;
    //orithresh = temp;

    }
    fprintf(stderr, "orithresh = %d\n", orithresh);

    while ( Vfread( &im, IVAL) ) {
        if ( im.type != VX_PBYTE ) {
              fprintf (stderr, "error: image not byte type\n");
              exit (1);
        }
    /*clear the histogram"*/
    for (i=0; i <256; i++) hist[0]=0;
		for(y=im.ylo; y<=im.yhi; y++)
		  for(x=im.xlo; x<=im.xhi; x++)
			hist[im.u[y][x]++;
      /*computer the threshold*/
    while(1){
      for(i = 0; i < orithresh; i++){
		sum1 += i * hist[i];
		count1 += hist[i];
      }
      for(i = orithresh; i <256; i++){
		sum2 += i*hist[i];
		count2 += hist[i];
      }

	 if(count1 == 0)
		avg1 = 0;
	 else
		avg1 = sum1 / count1; 
	 if(count2 == 0)
		avg2 = 0;
	 else      
		avg2 = sum2 / count2;
	 orithresh = (avg1 + avg2) / 2;
	 if(lasavg1 == avg1 && lasavg2 == avg2){
		fprintf(stderr, "avg1 = %d\n", avg1);
		fprintf(stderr, "avg2 = %d\n", avg2);
		break; 
	}
	 lasavg1 = avg1; 
	 lasavg2 = avg2;
	fprintf(stderr, "avg1 = %d\n", avg1);
	fprintf(stderr, "avg2 = %d\n", avg2);
	fprintf(stderr, "orithresh = %d\n", orithresh);
         
    }
 
    thresh = orithresh;
    fprintf(stderr, "thresh = %d\n", thresh);
  
	if(VFLAG)
		 fprintf(stderr, "thresh = %d\n", thresh);

	/* apply the threshold */
	for (y = im.ylo; y <= im.yhi; y++) {
		for (x = im.xlo; x <= im.xhi; x++) {
			 if (im.u[y][x] >= thresh) im.u[y][x] = 255;
			 else                      im.u[y][x] = 0;
		}
	}

	Vfwrite( &im, OVAL);
    } /* end of every frame section */
    exit(0);
}

