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

//define the function to sort the tuple of numbers and find the median
int find_median(int nums[27]){
	//sort the tuples
	int i, j;
	for (i = 1; i < 27; i++ ) {
		int j = i;
		int target = nums[i];
		for (j = i; j>=0; j--) {
			if(j > 0 && nums[j-1] > target)
				nums[j] = nums[j-1];
			else{
				nums[j] = target;
				break;
			}
		}
	}
	return nums[13];
}


int main(argc, argv)
int argc;
char *argv[];
{
	V3fstruct (im);
	V3fstruct (tm);
	int x,y,z;               /* index counters                 */
	int xx,yy,zz;            /* window index counters          */
	int median, count;
	int nums[27];
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
			count = 0;
             for (zz = -1; zz <= 1; zz++) {/* compute the function */
               for (yy = -1; yy <= 1; yy++) {
                 for (xx = -1; xx <= 1; xx++) {
					 nums[count] = tm.u[z + zz][y + yy][x + xx];
					 count++;	 
                 }   
               }   
             }
			 median = find_median(nums);
             im.u[z][y][x] = median;	 
        }
      }
   }
   V3fwrite (&im, OVAL);
   exit(0);
}
