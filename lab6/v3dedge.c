/****************************************************************/
/* Example VisX4 program v3df                                   */
/*            Compute function on a 3D image structure          */
/* Syntax:                                                      */
/*        v3df if=infile of=outfile [-v]                        */
/****************************************************************/

#include "VisXV4.h"          /* VisionX structure include file       */
#include "Vutil.h"           /* VisionX utility header files         */

VXparam_t par[] =            /* command line structure               */
{
{    "if=",    0,   " input file  v3dmean: compute local mean"},
{    "of=",    0,   " output file "},
{    "-v",     0,   " visible flag"},
{     0,       0,    0}
};
#define  IVAL   par[0].val
#define  OVAL   par[1].val
#define  VFLAG  par[2].val

int
main(argc, argv)
int argc;
char *argv[];
{
V3fstruct (im);
V3fstruct (tm);
int        x,y,z;               /* index counters                 */
int        xx,yy,zz;            /* window index counters          */
int        sum;
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

    for (z = im.zlo; z <= im.zhi; z++) {/* for all pixels */
      for (y = im.ylo; y <= im.yhi; y++) {
        for (x = im.xlo; x <= im.xhi; x++) {
			sum = false; 
             for (zz = -1; zz <= 1; zz++) {/* compute the function */
               for (yy = -1; yy <= 1; yy++) {
                 for (xx = -1; xx <= 1; xx++) {
					 if(tm.u[z + zz][y + yy][x + xx] == 0)
						 sum = ture;
                 }   
               }   
             }   
             if(tm.u[z][y][x] != 0 && sum == true) 
				 im.u[z][y][x] = 255;
			 else if(tm.u[z][y][x] != 0 && sum == false)
				 im.u[z][y][x] = 128;
			 else 
				 im.u[z][y][x] = 0;	 
        }
      }
   }
   V3fwrite (&im, OVAL);
   exit(0);
}
