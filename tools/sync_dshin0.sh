#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR=$DIR/..

set -ex

cd ${PROJ_DIR}

rsync -atvur --info=progress2,name0 ~/git/3D-R2N2 beta.ics:~/git/
rsync -atvur --info=progress2,name0 ~/git/mvshape/python beta.ics:~/git/mvshape
echo "Synced python"
rsync -atvur --info=progress2,name0 ~/git/mvshape beta.ics:~/git/
rsync -atvur --info=progress2,name0 ~/Dropbox/git/dshinpy beta.ics:~/Dropbox/git/


REMOTE_HOST=dshin0
rsync -atvurXxAH --exclude="data" --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" ~/git/RenderForCNN ${REMOTE_HOST}:~/git/
rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" ~/git/mvshape ${REMOTE_HOST}:~/git/
rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" ~/Dropbox/git/dshinpy ${REMOTE_HOST}:~/Dropbox/git/
rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" ~/usr ${REMOTE_HOST}:~/



#rsync -atvurz --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" ${REMOTE_HOST}:/data/syn_images_final /data/


rsync -atvur --info=progress2,name0 /data/mvshape/out/splits beta.ics:~/data/mvshape/out/



#REMOTE_HOST=rhea.ics
#rsync -atvurXxAH --exclude="data" --exclude="datasets" --info=progress2 -e "ssh -T -c arcfour -o Compression=no -x" ~/git/RenderForCNN ${REMOTE_HOST}:/extra/titansc0/daeyun_temp/home/git
#rsync -atvurXxAH --info=progress2 -e "ssh -T -c arcfour -o Compression=no -x" ~/git/mvshape ${REMOTE_HOST}:/extra/titansc0/daeyun_temp/home/git/
#rsync -atvurXxAH --info=progress2 -e "ssh -T -c arcfour -o Compression=no -x" ~/Dropbox/git/dshinpy ${REMOTE_HOST}:/extra/titansc0/daeyun_temp/home/git/

#rsync -atvurXxAH --info=progress2 -e "ssh -T -c arcfour -o Compression=no -x" /media/daeyun/Research\ Data/ShapeNetCore.v1.zip ${REMOTE_HOST}:/scratch/daeyuns/

#rsync -atvurXxAH --info=progress2 -e "ssh -T -c arcfour -o Compression=no -x" \
    #/data/render_for_cnn/data/detection_results \
    #/data/render_for_cnn/data/truncation_distribution \
    #/data/render_for_cnn/data/truncation_statistics \
    #/data/render_for_cnn/data/view_distribution \
    #/data/render_for_cnn/data/view_statistics \
    #${REMOTE_HOST}:~/scratch/render_for_cnn/data/

#rsync -atvurXxAH --info=progress2 -e "ssh -T -c arcfour -o Compression=no -x" \
    #/data/mvshape/mesh/shapenetcore/v1/02691156 \
    #/data/mvshape/mesh/shapenetcore/v1/02834778 \
    #/data/mvshape/mesh/shapenetcore/v1/02858304 \
    #/data/mvshape/mesh/shapenetcore/v1/02876657 \
    #/data/mvshape/mesh/shapenetcore/v1/02924116 \
    #/data/mvshape/mesh/shapenetcore/v1/02958343 \
    #/data/mvshape/mesh/shapenetcore/v1/03001627 \
    #/data/mvshape/mesh/shapenetcore/v1/03211117 \
    #/data/mvshape/mesh/shapenetcore/v1/03790512 \
    #/data/mvshape/mesh/shapenetcore/v1/04256520 \
    #/data/mvshape/mesh/shapenetcore/v1/04379243 \
    #/data/mvshape/mesh/shapenetcore/v1/04468005 \
    #${REMOTE_HOST}:~/scratch/v1







#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #dshin0:/home/daeyun/git/RenderForCNN/data/tmp_view_Y_P90V/tmpFLVLJc \
    #/tmp/hi/

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #dshin0:/data/render_for_cnn/data/syn_images/02691156 \
    #/data/render_for_cnn/data/syn_images/

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #dshin0:/data/render_for_cnn/data/syn_images/02858304 \
    #dshin0:/data/render_for_cnn/data/syn_images/02876657 \
    #dshin0:/data/render_for_cnn/data/syn_images/02924116 \
    #dshin0:/data/render_for_cnn/data/syn_images/04379243 \
    #/data/render_for_cnn/data/syn_images/

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #/data/render_for_cnn/data/syn_images/ \
    #dshin0:/data/render_for_cnn/data/syn_images

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #dshin0:/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/02858304 \
    #dshin0:/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/02876657 \
    #dshin0:/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/02924116 \
    #dshin0:/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/04379243 \
    #/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/ \
    #dshin0:/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #/data/render_for_cnn/data/syn_images/ \
    #dshin0:/data/render_for_cnn/data/syn_images/

#rsync -atvurXxAH --info=progress2,name0 -e "ssh -T -c arcfour -o Compression=no -x" \
    #/data/mvshape/mesh/shapenetcore/v1 \
    #dshin0:/data/shapenetcore/ShapeNetCore.v1

