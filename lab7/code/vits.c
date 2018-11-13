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
{  "th=",    0,   " the initial threshold (default the average of all pixel values)"},
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
    int thresh;                    /* threshold                      */
    int orithresh = 100;              /* the original threshold   */
    int temp;  
    int count = 0; 
                                  
			     
    VXparse(&argc, &argv, par);    /* parse the command line         */

    if (TVAL) orithresh = atoi(TVAL);  /* if d= was specified, get value */
    if (orithresh < 0 || orithresh > 255) {
	fprintf(stderr, "d= must be between 0 and 255\nUsing d=10\n");
        orithresh = 100;

    }
	fprintf(stderr, "orithresh = %d\n", orithresh);

    while ( Vfread( &im, IVAL) ) {
        if ( im.type != VX_PBYTE ) {
              fprintf (stderr, "error: image not byte type\n");
              exit (1);
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
