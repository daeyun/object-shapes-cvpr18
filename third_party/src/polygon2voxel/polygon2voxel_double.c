// (c) 2009, Dirk-Jan Kroon, BSD 2-Clause
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define min(X,Y) ((X) < (Y) ? (X) : (Y)) 
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

int mindex3(int x, int y, int z, int sizx, int sizy, int sizz, int wrap) 
{ 
    int index;
 	if(wrap==1)
	{
        /* Positive modules */
        x=(x % sizx + sizx) % sizx;
        y=(y % sizy + sizy) % sizy;
        z=(z % sizz + sizz) % sizz;
    }
    else if(wrap>1)
    {
        /* Clamp */
        x=max(x,0);
        y=max(y,0); 
        z=max(z,0);
        x=min(x,sizx-1);
        y=min(y,sizy-1);
        z=min(z,sizz-1);
    }
    index=z*sizx*sizy+y*sizx+x;
    return index;
}

uint8_t *draw_or_split(uint8_t *Volume,double AX,double AY,double AZ,double BX,double BY,double BZ,double CX,double CY,double CZ, const int *VolumeSize, int wrap)
{
    bool checkA=0, checkB=0, checkC=0;
    bool check1, check2, check3, check4, check5, check6;

    double dist1, dist2, dist3, maxdist;
    double DX,DY,DZ;
    /* Check if vertices outside */
    if(wrap==0)
	{
        checkA=(AX<0)||(AY<0)||(AZ<0)||(AX>(VolumeSize[0]-1))||(AY>(VolumeSize[1]-1))||(AZ>(VolumeSize[2]-1));
        checkB=(BX<0)||(BY<0)||(BZ<0)||(BX>(VolumeSize[0]-1))||(BY>(VolumeSize[1]-1))||(BZ>(VolumeSize[2]-1));
        checkC=(CX<0)||(CY<0)||(CZ<0)||(CX>(VolumeSize[0]-1))||(CY>(VolumeSize[1]-1))||(CZ>(VolumeSize[2]-1));

        check1=(AX<0)&&(BX<0)&&(CX<0);
		check2=(AY<0)&&(BY<0)&&(CY<0);
		check3=(AZ<0)&&(BZ<0)&&(CZ<0);
		check4=(AX>(VolumeSize[0]-1))&&(BX>(VolumeSize[0]-1))&&(CX>(VolumeSize[0]-1));
		check5=(AY>(VolumeSize[1]-1))&&(BY>(VolumeSize[1]-1))&&(CY>(VolumeSize[1]-1));
		check6=(AZ>(VolumeSize[2]-1))&&(BZ>(VolumeSize[2]-1))&&(CZ>(VolumeSize[2]-1));
		
		/* Return if all vertices outside, on the same side */
		if(check1||check2||check3||check4||check5||check6)
		{
			return Volume;
		}
	}
	
    dist1=(AX-BX)*(AX-BX)+(AY-BY)*(AY-BY)+(AZ-BZ)*(AZ-BZ);
    dist2=(CX-BX)*(CX-BX)+(CY-BY)*(CY-BY)+(CZ-BZ)*(CZ-BZ);
    dist3=(AX-CX)*(AX-CX)+(AY-CY)*(AY-CY)+(AZ-CZ)*(AZ-CZ);
    if(dist1>dist2)
    {
        if(dist1>dist3)
        {
            maxdist=dist1;
            if(maxdist>0.5)
            {
                DX=(AX+BX)/2; DY=(AY+BY)/2; DZ=(AZ+BZ)/2;
                Volume=draw_or_split(Volume,DX,DY,DZ,BX,BY,BZ,CX,CY,CZ,VolumeSize, wrap);
                Volume=draw_or_split(Volume,AX,AY,AZ,DX,DY,DZ,CX,CY,CZ,VolumeSize, wrap);
            }  
        }
        else
        {
            maxdist=dist3;
            if(maxdist>0.5)
            {
                DX=(AX+CX)/2; DY=(AY+CY)/2; DZ=(AZ+CZ)/2;
                Volume=draw_or_split(Volume,DX,DY,DZ,BX,BY,BZ,CX,CY,CZ,VolumeSize, wrap);
                Volume=draw_or_split(Volume,AX,AY,AZ,BX,BY,BZ,DX,DY,DZ,VolumeSize, wrap);
            }  

        }
    }
    else
    {
        if(dist2>dist3)
        {
            maxdist=dist2;
            DX=(CX+BX)/2; DY=(CY+BY)/2; DZ=(CZ+BZ)/2;
            if(maxdist>0.5)
            {
                Volume=draw_or_split(Volume,AX,AY,AZ,DX,DY,DZ,CX,CY,CZ,VolumeSize, wrap);
                Volume=draw_or_split(Volume,AX,AY,AZ,BX,BY,BZ,DX,DY,DZ,VolumeSize, wrap);
            }  
        }
        else
        {
            maxdist=dist3;
            if(maxdist>0.5)
            {
                DX=(AX+CX)/2; DY=(AY+CY)/2; DZ=(AZ+CZ)/2;
                Volume=draw_or_split(Volume,DX,DY,DZ,BX,BY,BZ,CX,CY,CZ,VolumeSize, wrap);
                Volume=draw_or_split(Volume,AX,AY,AZ,BX,BY,BZ,DX,DY,DZ,VolumeSize, wrap);
            }  

        }
    }
    if(wrap==0)
    {      
        if(checkA==false)
        {
            Volume[mindex3((int)(AX+0.5),(int)(AY+0.5), (int)(AZ+0.5), VolumeSize[0], VolumeSize[1], VolumeSize[2], wrap)]=1;
        }
        if(checkB==false)
        {
            Volume[mindex3((int)(BX+0.5),(int)(BY+0.5), (int)(BZ+0.5), VolumeSize[0], VolumeSize[1], VolumeSize[2], wrap)]=1;
        }
        if(checkC==false)
        {
            Volume[mindex3((int)(CX+0.5),(int)(CY+0.5), (int)(CZ+0.5), VolumeSize[0], VolumeSize[1], VolumeSize[2], wrap)]=1;
        }
    }
    else
    {
         Volume[mindex3((int)(AX+0.5),(int)(AY+0.5), (int)(AZ+0.5), VolumeSize[0], VolumeSize[1], VolumeSize[2], wrap)]=1;
         Volume[mindex3((int)(BX+0.5),(int)(BY+0.5), (int)(BZ+0.5), VolumeSize[0], VolumeSize[1], VolumeSize[2], wrap)]=1;
         Volume[mindex3((int)(CX+0.5),(int)(CY+0.5), (int)(CZ+0.5), VolumeSize[0], VolumeSize[1], VolumeSize[2], wrap)]=1;
    }
    return Volume;
}

void polygon2voxel(const int32_t *FacesA,
                   const int32_t *FacesB,
                   const int32_t *FacesC,
                   const int32_t num_faces,
                   const double *VerticesX,
                   const double *VerticesY,
                   const double *VerticesZ,
                   const int32_t *VolumeDims,
                   int32_t wrap,
                   uint8_t *OutVolume) {
    double AX,AY,AZ;
    double BX,BY,BZ;
    double CX,CY,CZ;
    int i;

    // OutVolume = (uint8_t *) calloc(VolumeDims[0] * VolumeDims[1] * VolumeDims[2], sizeof(uint8_t));

    for (i=0; i<num_faces; i++)
    {
        // NOTE(daeyun): Unlike in the original version, the faces are 0-indexed.
        // Voxel coordinates also start at 0 instead of 1.
        AX=VerticesX[FacesA[i]];
        AY=VerticesY[FacesA[i]];
        AZ=VerticesZ[FacesA[i]];
        BX=VerticesX[FacesB[i]];
        BY=VerticesY[FacesB[i]];
        BZ=VerticesZ[FacesB[i]];
        CX=VerticesX[FacesC[i]];
        CY=VerticesY[FacesC[i]];
        CZ=VerticesZ[FacesC[i]];
        OutVolume=draw_or_split(OutVolume,AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,VolumeDims,wrap);
    }
}
