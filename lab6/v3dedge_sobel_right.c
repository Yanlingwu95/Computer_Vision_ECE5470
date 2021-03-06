/****************************************************************/
/* Example VisX4 program v3df                                   */
/*            Compute function on a 3D image structure          */
/* Syntax:                                                      */
/*        v3df if=infile of=outfile [-v]                        */
/****************************************************************/

#include "VisXV4.h"          /* VisionX structure include file       */
#include "Vutil.h"           /* VisionX utility header files         */
#include <math.h>

VXparam_t par[] =            /* command line structure               */
{
{    "if=",    0,   " input file  v3dmean: compute local mean"},
{    "of=",    0,   " output file "},
{    "-v",     0,   " visible flag"},
{    "th",     0,   "threshold"},
{     0,       0,    0}
};
#define  IVAL   par[0].val
#define  OVAL   par[1].val
#define  VFLAG  par[2].val
#define  TFLAG  par[3].val

int
main(argc, argv)
int argc;
char *argv[];
{
V3fstruct (im);
V3fstruct (tm);
int        x,y,z;               /* index counters                 */
int        xx,yy,zz;            /* window index counters          */
int        sum, grad;
float gradth;
int th;
int gy[3][3][3], gx[3][3][3], gz[3][3][3];
int hx[3]={1,2,1}, hy[3]={1,2,1},hz[3]={1,2,1};
int hpx[3]={1,0,-1},hpy[3]={1,0,-1},hpz[3]={1,0,-1};
int sumx, sumy, sumz;
    VXparse(&argc, &argv, par); /* parse the command line         */

    V3fread( &im, IVAL);        /* read 3D image                  */
    if ( im.type != VX_PBYTE || im.chan != 1) { /* check  format  */
       fprintf (stderr, "image not byte type or single channel\n");
       exit (1);
    }   
   
    V3fembed(&tm, &im, 1,1,1,1,1,1); /* temp image copy with border */
    if(VFLAG){
       fprintf(stderr,"bbx is %f %f %f %f %f %f\n", im.bbx[0],
                 im.bbx[1],im.bbx[2],im.bbx[3],im.bbx[4],im.bbx[5]);
    }
    th = (TFLAG ? atoi(TFLAG) : 50);
	for(xx=0;xx<=2;xx++){ //build the kernel
		for(yy=0;yy<=2;yy++){
			for(zz=0;zz<=2;zz++){
				gx[xx][yy][zz]=hpx[xx]*hy[yy]*hz[zz];
				gy[xx][yy][zz]=hx[xx]*hpy[yy]*hz[zz];
				gz[xx][yy][zz]=hx[xx]*hy[yy]*hpz[zz];
				}
		}
	}
	
    for (z = im.zlo; z <= im.zhi; z++) {/* for all pixels */
      for (y = im.ylo; y <= im.yhi; y++) {
        for (x = im.xlo; x <= im.xhi; x++) {
			sumx = 0; sumy = 0; sumz = 0;
             for (zz = -1; zz <= 1; zz++) {/* compute the function */
               for (yy = -1; yy <= 1; yy++) {
                 for (xx = -1; xx <= 1; xx++) {
					 sumx += gx[zz+1][yy+1][zz+1]*tm.u[z-zz][y-yy][x-xx];
					 sumy += gy[zz+1][yy+1][zz+1]*tm.u[z-zz][y-yy][x-xx];
					 sumz += gz[zz+1][yy+1][zz+1]*tm.u[z-zz][y-yy][x-xx];
					 
                 }   
               }   
             }   
			 sumx /= 27;
			 sumy /= 27;
			 sumz /= 27;
			 gradth = sqrt(sumx*sumx+ sumy*sumy+sumz*sumz);
			 if(gradth > th)
			 	im.u[z][y][x]=255;
			else 
				im.u[z][y][x] = 0; //threshold at 50   
        }
      }
   }
   V3fwrite (&im, OVAL);
   exit(0);
}
