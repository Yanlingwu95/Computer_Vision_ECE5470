/*********************************************************************/
/* vssum   Compute local 1x1xn mean using the buffer method          */
/*********************************************************************/

#include "VisXV4.h"          /* VisionX structure include file       */
#include "Vutil.h"           /* VisionX utility header files         */

VXparam_t par[] =            /* command line structure               */
{
{    "if=",    0,   " input file vssum: compute temporal mean"},
{    "of=",    0,   " output file "},
{    "n=",     0,   " number of frames "},
{    "th=",    0,   " the value of threshold "},
{     0,       0,    0}
};
#define  IVAL   par[0].val
#define  OVAL   par[1].val
#define  NVAL   par[2].val
#define  TVAL   par[3].val

int main(argc, argv)
int argc;
char *argv[];
{
V3fstruct (im);
V3fstruct (tm);
int        x,y,z;           /* index counters                 */
int        n;               /* Number of frames to average    */
int        th;

    VXparse(&argc, &argv, par); /* parse the command line    */

    n = (NVAL ? atoi(NVAL) : 2); /* read n, default is n=1   */

    while (Vbfread( &im, IVAL, n)) {
	if ( im.type != VX_PBYTE || im.chan != 1) { /* check format  */
           fprintf (stderr, "image not byte type\n");
           exit (1);
        }
        for (y = im.ylo; y <= im.yhi; y++) {
           for (x = im.xlo; x <= im.xhi; x++) {
              z = im.zlo;
			  if(th < abs(im.u[z][y][x] - im.u[z+1][y][x]))
				  im.u[z][y][x] = 255;
			  else 
				  im.u[z][y][x] = 128;
            }
        }
        V3fwrite (&im, OVAL); /* write the oldest frame */
    }
    exit(0);
}
