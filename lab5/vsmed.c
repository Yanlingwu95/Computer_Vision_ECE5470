/*********************************************************************/
/* vssum   Compute local 1x1xn mean using the buffer method          */
/* Yanling Wu														 */
/* NetID: yw996  													 */
/* Lab5 															 */
/*********************************************************************/

#include "VisXV4.h"          /* VisionX structure include file       */
#include "Vutil.h"           /* VisionX utility header files         */

VXparam_t par[] =            /* command line structure               */
{
{    "if=",    0,   " input file vssum: compute temporal mean"},
{    "of=",    0,   " output file "},
{    "n=",     0,   " number of frames "},
{     0,       0,    0}
};
#define  IVAL   par[0].val
#define  OVAL   par[1].val
#define  NVAL   par[2].val

int
main(argc, argv)
int argc;
char *argv[];
{
V3fstruct (im);
V3fstruct (tm);
int        x,y,z;           /* index counters                 */
int        n;               /* Number of frames to average    */
int        val1, val2;

    VXparse(&argc, &argv, par); /* parse the command line    */

    n = (NVAL ? atoi(NVAL) : 3); /* read n, default is n=1   */

    while (Vbfread( &im, IVAL, n)) {
	if ( im.type != VX_PBYTE || im.chan != 1) { /* check format  */
           fprintf (stderr, "image not byte type\n");
           exit (1);
        }
        for (y = im.ylo; y <= im.yhi; y++) {
           for (x = im.xlo; x <= im.xhi; x++) {
				val1 = 0; 
				val2 = 0;
				z = im.zlo;
				if(im.u[z][y][x] >= im.u[z+1][y][x]){
				   val1 = im.u[z][y][x];
				   val2 = im.u[z+1][y][x];
				}
				else{
				   val1 = im.u[z+1][y][x];
				   val2 = im.u[z][y][x];
				}
				if(val1 > im.u[z+2][y][x]){
					if(val2 >= im.u[z+2][y][x])
					   im.u[0][y][x] = val2;
				   else 
					   im.u[0][y][x] = im.u[z+2][y][x]; 
				}
				else{
				   im.u[0][y][x] = val1;
				}
            }
        }
        V3fwrite (&im, OVAL); /* write the oldest frame */
    }
    exit(0);
}
