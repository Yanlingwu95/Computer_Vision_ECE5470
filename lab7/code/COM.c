/*******************************************************************/
/* vcorner return the left lower corner of a byte image            */
/*******************************************************************/
#include "VisXV4.h"	/* VisionX structure include file    */
#include "Vutil.h"	/* VisionX utility header files      */
VXparam_t par[] =             /* command line structure            */
{ /* prefix, value,   description                         */
{    "if=",    0,   " input file  vcorner: copy lower left corner"},
{    "of=",    0,   " output file "},
{     0,       0,   0}  /* list termination */
};
#define IVAL par[0].val
#define OVAL par[1].val
int main(argc, argv)
int argc;
char *argv[];
{
	
	Vfstruct(im);                      /* i/o image structure          */
	Vfstruct(tm);                      /* temp image structure         */
	int y,x;                     /* index counters               */
	int s;          /* corner image size             */
	float bbx[6];   /* bounding box for corner image */
	int area = 0, x_move = 0, y_move = 0;
	int mx = 0, my = 0;
	VXparse(&argc, &argv, par);       /* parse the command line       */
	/* create VX image for result */
	s = 28;  /* set size of result image */
	bbx[0] = bbx[2] = bbx[4] = bbx[5] = 0.0;
	bbx[1] = bbx[3] = s;
	Vfnewim(&tm, VX_PBYTE, bbx, 1);
	while ( Vfread(&im, IVAL) ) {     /* read image file              */
		if ( im.type != VX_PBYTE ) {    /* check image format           */
			fprintf(stderr, "vcorner: no byte image data in input file\n");
			exit(-1);
		}
		/* check that the input image is large enough */
	/*	if ( (im.xhi - im.xlo ) < (tm.xhi - tm.xlo) || (im.yhi - im.ylo ) < (tm.yhi - tm.ylo) ) {
			fprintf(stderr, "vcorner: input image too small\n");
			exit(-1);
		}*/
		//compute the original COM position
		for (y = im.ylo; y <= im.yhi; y++) {
			for(x = im.xlo; x <= im.xhi; x++) {
				if (im.u[y][x] != 0) {
					area += 1;
					mx += x;
					my += y;
				}
			}
		}
		fprintf(stderr, "xmean=%d,ymean=%d\n", mx/area,my/area);

		x_move = 14 - mx / area ;
		y_move = 14 - my / area;
		fprintf(stderr, "x_move=%d, y_move=%d\n",x_move,y_move);
		
		/* this loop assumes tm.xlo and tm.ylo are 0 but not im.xlo and im.ylo */
		for (y = im.ylo; y <= im.yhi; y++) {  /* compute the function */
			for(x = im.xlo; x <= im.xhi; x++)  { 
				tm.u[y+y_move][x+x_move] = im.u[y][x];
			}
		}
		Vfwrite(&tm, OVAL);             /* write image file                */
	}
	exit(0);
}
