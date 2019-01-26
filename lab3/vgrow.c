/*******************************************************************/
/*vgrow      Compute label operation on a single byte image      */
/*******************************************************************/
/*@copyright													   */
/*Name: Yanling Wu                                                 */
/*NetID: yw996                                                     */
/*LAB Number: LAB 2                                                */ 
/*******************************************************************/

#include "VisXV4.h"           /* VisionX structure include file    */
#include "Vutil.h"            /* VisionX utility header files      */

VXparam_t par[] =             /* command line structure            */
{ /* prefix, value,   description                         */   
{    "if=",    0,   " input file  vtemp: local max filter "},
{    "of=",    0,   " output file "},
{    "r=",     0,   " set the region pixel range"},
{    "-p",     0,   " select the labeling scheme"},
{     0,       0,   0}  /* list termination */
};
#define  IVAL   par[0].val
#define  OVAL   par[1].val
#define  RVAL   par[2].val
#define  PVAL   par[3].val

void setlabel(int ,int ,int );
Vfstruct (im);                      /* i/o image structure          */
Vfstruct (tm);                      /* temp image structure         */
int range;                          /*global varibale            */
int first;

main(argc, argv)
int argc;
char *argv[];
{
	
  int y,x;                          /* index counters               */
  VXparse(&argc, &argv, par);       /* parse the command line       */
  Vfread(&im, IVAL);                /* read image file              */
  Vfembed(&tm, &im, 1,1,1,1);       /* image structure with border  */
  if (RVAL) 
	  range = atoi(TVAL);     /* if t= was specified, get value */
  else 
	  range = 10;                  /*set default range value     */
  if ( im.type != VX_PBYTE ) {       /* check image format           */
     fprintf(stderr, "vtemp: no byte image data in input file\n");
     exit(-1);
  }
  //set all pixels (labels) to 0
  for (y = im.ylo ; y <= im.yhi ; y++) {
	     for (x = im.xlo; x <= im.xhi; x++)  
			im.u[y][x] = 0;	
	}
  int n =1;
/*  Scan the image to label every unlabeled point */
  for (y = im.ylo ; y <= im.yhi ; y++) {
     for (x = im.xlo; x <= im.xhi; x++)  {
		if(tm.u[y][x] != 0 && im.u[y][x] == 0){
			first = tm.u[y][x];
			if (PVAL) 
				setlabel(x,y,first);
			else {
				setlabel(x,y,n);
				n++;
				if(n > 255) 
					n = n % 256;  /*in case the label over the ranger of pixel value */
			}
		}
     }
   }

   Vfwrite(&im, OVAL);             /* write image file  */
   exit(0);
}
/* setlabel function   */
void setlabel(int x,int y,int L){
	im.u[y][x] = L;
	if ( tm.u[y+1][x] != 0 && im.u[y+1][x] == 0 && tm.u[y+1][x] - first < range ) 
		setlabel(x,y+1,L);
	if( tm.u[y-1][x] != 0 && im.u[y-1][x] == 0 && tm.u[y-1][x] - first < range )
		setlabel(x,y-1,L);
	if(tm.u[y][x-1] != 0 && im.u[y][x-1] == 0 && tm.u[y][x-1] - first < range )
		setlabel(x-1,y,L);
	if(tm.u[y][x+1] != 0 && im.u[y][x+1] == 0 && tm.u[y][x+1] - first < range )
		setlabel(x+1,y,L);

	
}