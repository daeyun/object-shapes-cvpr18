import shutil
import os
from os import path
import uuid
import numpy as np
from mvshape import db_model as dbm
from mvshape.db_model import *


def unique(item):
    try:
        assert len(item) == 1, len(item)
    except TypeError:
        return item
    return item[0]


def main():
    cached_renderings = {}

    target_db = '/data/mvshape/database_wip/shrec12.sqlite'

    shutil.copy('/data/mvshape/database_backup/shrec12.sqlite', target_db)

    dbm.init(target_db)

    # global camera
    camera = Camera(
        fov=20.0,
        is_orthographic=True,
        lookat=np.array([0, 0, 0]),
        position_xyz=np.array([15, 0, 0]),
        up=np.array([0, 0, 1]),
        scale=0.4,
    )
    camera.save()

    assert camera.id == 101

    # find all existing examples. we want to duplicate them and give new parameters.
    examples = Example.select()

    depth = RenderingType.select().where(RenderingType.name == 'depth')[0]
    normal = RenderingType.select().where(RenderingType.name == 'normal')[0]
    rgb = RenderingType.select().where(RenderingType.name == 'rgb')[0]
    voxels = RenderingType.select().where(RenderingType.name == 'voxels')[0]

    assert depth.name == 'depth'
    assert normal.name == 'normal'
    assert rgb.name == 'rgb'
    assert voxels.name == 'voxels'

    object_centered = unique(Tag.select().where(Tag.name == 'object_centered'))
    print(object_centered.id)

    def make_new_example(example):
        # Make a new example that corresponds to this one.
        new_example = Example()
        new_example.save()
        print('base example id ', example.id, ', new example id ', new_example.id)

        # ---------------------------------

        # find the dataset this example is associated with.
        dataset = Dataset.select().join(ExampleDataset).join(Example).where(Example.id == example.id)
        assert len(dataset) == 1, len(dataset)
        assert dataset[0].name == 'shrec12'

        # create new association, save.
        ExampleDataset(
            example=new_example,
            dataset=dataset,
        ).save()

        # ---------------------------------

        # find the renderings this example is associated with.
        renderings = ObjectRendering.select().join(ExampleObjectRendering).join(Example).where(Example.id == example.id)

        # rgb, depth, mv_rgb, mv_depth, mv_normals, voxels.
        assert len(renderings) == 6

        # we want to keep single view rgb and depth the same.
        # but create new mv_rgb, mv_depth, mv_normals, voxels.
        # but only once per object. save those 4 in a dict object.id->renderings, and reuse later.

        single_depth = unique(renderings.where((ObjectRendering.type == depth) &
                                               (ObjectRendering.set_size == 1)))
        single_rgb = unique(renderings.where((ObjectRendering.type == rgb) &
                                             (ObjectRendering.set_size == 1)))
        mv_depth = unique(renderings.where((ObjectRendering.type == depth) &
                                           (ObjectRendering.set_size == 6)))
        mv_normal = unique(renderings.where((ObjectRendering.type == normal) &
                                            (ObjectRendering.set_size == 6)))
        mv_rgb = unique(renderings.where((ObjectRendering.type == rgb) &
                                         (ObjectRendering.set_size == 6)))
        vox = unique(renderings.where(ObjectRendering.type == voxels))

        # random string. we use the same one for the 4 new renderings, but this is just for convenience and
        # it wouldn't make any difference if they were distinct.
        name = str(uuid.uuid4())[:18]

        def new_rendering(rendering, kind):
            def make_new(r):
                new_r = ObjectRendering()
                new_r.type = r.type
                new_r.object = r.object
                new_r.filename = path.join(path.dirname(r.filename), "{}.bin".format(name))
                new_r.resolution = r.resolution
                new_r.num_channels = r.num_channels
                new_r.set_size = r.set_size
                new_r.is_normalized = r.is_normalized
                new_r.camera = camera  # object_centered camera

                # make sure file name doesnt already exist.
                assert not os.path.exists(path.join('/data/mvshape', new_r.filename[1:]))

                new_r.save()

                return new_r

            key = (rendering.object.id, kind)

            if key not in cached_renderings:
                cached_renderings[key] = make_new(rendering)
            return cached_renderings[key]

        new_mv_depth = new_rendering(mv_depth, 'depth')
        new_mv_normal = new_rendering(mv_normal, 'normal')
        new_mv_rgb = new_rendering(mv_rgb, 'rgb')
        new_vox = new_rendering(vox, 'voxels')

        ExampleObjectRendering(
            example=new_example,
            rendering=single_depth
        ).save()
        ExampleObjectRendering(
            example=new_example,
            rendering=single_rgb
        ).save()
        ExampleObjectRendering(
            example=new_example,
            rendering=new_mv_depth
        ).save()
        ExampleObjectRendering(
            example=new_example,
            rendering=new_mv_normal
        ).save()
        ExampleObjectRendering(
            example=new_example,
            rendering=new_mv_rgb
        ).save()
        ExampleObjectRendering(
            example=new_example,
            rendering=new_vox
        ).save()

        # ---------------------------------

        # one of test, validation, train, etc. use the same without creating a new one.
        split = unique(Split.select().join(ExampleSplit).join(Example).where(Example.id == example.id))

        ExampleSplit(
            example=new_example,
            split=split,
        ).save()

        # ---------------------------------

        # all tags associated with this example. we'll replace viewer-centered with object centered.
        tags = Tag.select().join(ExampleTag).join(Example).where(Example.id == example.id)

        new_tags = []
        for tag in tags:
            if tag.name != 'viewer_centered':
                new_tags.append(tag)

        new_tags.append(object_centered)
        assert len(new_tags) == len(tags)

        for item in new_tags:
            ExampleTag(
                example=new_example,
                tag=item,
            ).save()

    num_examples = len(examples)
    with dbm.db.transaction() as txn:
        i = 0
        for e in examples:
            make_new_example(e)

            if i % 1000 == 0:
                txn.commit()
            i += 1
        txn.commit()

    dbm.db.commit()

    assert num_examples * 2 == len(Example.select())


if __name__ == '__main__':
    main()
