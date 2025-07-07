
# Re-Identification (ReID)

## Overview

Object trackers like ByteTrack can be used to track visible objects frame‐to‐frame, 
but they rely on the assumption that an object's appearance and location change 
smoothly over time. If a person goes behind a building or is briefly hidden
by another passerby, the tracker can lose that objects identity. When that same 
person reemerges, the tracker often treats them as a new object, assigning a new ID.
This makes analyzing a persons complete path through a scene difficult
or makes counting unique objects much harder.

Re-Identification (ReID) models help solve this problem by using embedding features 
which encode an object into a fixed length vector that captures distinctive
patterns, shapes, or other visual signatures.  When an object disappears and 
then reappears you can compare the newly detected objects embedding against a list of
past objects. If the similarity (using Cosine or Euclidean distance) 
exceeds a chosen threshold, you can confidently link the new detection back to the 
original track ID.


## Datasets

The [OSNet model](https://paperswithcode.com/paper/omni-scale-feature-learning-for-person-re) is 
lite weight and provides good accuracy for reidentification tasks, however
it must be trained using a dataset to identify specific object classes.

This example uses the [Market1501](https://paperswithcode.com/dataset/market-1501) 
dataset trained for reidentifying people.

To support other object classifications such as Vehicles, Faces, or Animals, you
will need to source and train these accordingly.


## Occlusion Example

In the [people walking video](https://github.com/swdee/go-rknnlite-data/raw/master/people-walking.mp4) 
a lady wearing a CK branded jacket starts 
in the beginning of the scene and becomes occluded by passersby.  When she reappears Bytetrack
detects them as a new person.

![CK Lady](https://github.com/swdee/go-rknnlite-data/raw/master/docimg/reid-ck-lady-movement.jpg)



## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```


Command line Usage.
```
$ go run reid.go -h

Usage of /tmp/go-build147978858/b001/exe/reid:
  -d string
        Data file containing object co-ordinates (default "../data/reid-objects.dat")
  -e float
        The Euclidean distance [0.0-1.0], a value less than defines a match (default 0.51)
  -i string
        Image file to run inference on (default "../data/reid-walking.jpg")
  -m string
        RKNN compiled model file (default "../data/models/rk3588/osnet-market1501-batch8-rk3588.rknn")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```

Run the ReID example on rk3588 or replace with your Platform model.
```
cd example/reid/
go run reid.go -p rk3588
```


This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 1
Input tensors:
  index=0, name=input, n_dims=4, dims=[8, 256, 128, 3], n_elems=786432, size=786432, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-14, scale=0.018658
Output tensors:
  index=0, name=output, n_dims=2, dims=[8, 512, 0, 0], n_elems=4096, size=4096, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.018782
Comparing object 0 at (0,0,134,361)
  Object 0 at (0,0,134,361) has euclidean distance: 0.000000 (same person)
  Object 1 at (134,0,251,325) has euclidean distance: 0.423271 (same person)
  Object 2 at (251,0,326,208) has euclidean distance: 0.465061 (same person)
  Object 3 at (326,0,394,187) has euclidean distance: 0.445583 (same person)
Comparing object 1 at (394,0,513,357)
  Object 0 at (0,0,134,361) has euclidean distance: 0.781510 (different person)
  Object 1 at (134,0,251,325) has euclidean distance: 0.801649 (different person)
  Object 2 at (251,0,326,208) has euclidean distance: 0.680299 (different person)
  Object 3 at (326,0,394,187) has euclidean distance: 0.686542 (different person)
Comparing object 2 at (513,0,588,246)
  Object 0 at (0,0,134,361) has euclidean distance: 0.860921 (different person)
  Object 1 at (134,0,251,325) has euclidean distance: 0.873663 (different person)
  Object 2 at (251,0,326,208) has euclidean distance: 0.870753 (different person)
  Object 3 at (326,0,394,187) has euclidean distance: 0.820761 (different person)
Comparing object 3 at (588,0,728,360)
  Object 0 at (0,0,134,361) has euclidean distance: 0.762738 (different person)
  Object 1 at (134,0,251,325) has euclidean distance: 0.800668 (different person)
  Object 2 at (251,0,326,208) has euclidean distance: 0.763694 (different person)
  Object 3 at (326,0,394,187) has euclidean distance: 0.769597 (different person)
Model first run speed: batch preparation=3.900093ms, inference=47.935686ms, post processing=262.203µs, total time=52.097982ms
done
```

### Docker

To run the ReID example using the prebuilt docker image, make sure the data files have been downloaded first,
then run.
```
# from project root directory

docker run --rm \
  --device /dev/dri:/dev/dri \
  -v "$(pwd):/go/src/app" \
  -v "$(pwd)/example/data:/go/src/data" \
  -v "/usr/include/rknn_api.h:/usr/include/rknn_api.h" \
  -v "/usr/lib/librknnrt.so:/usr/lib/librknnrt.so" \
  -w /go/src/app \
  swdee/go-rknnlite:latest \
  go run ./example/reid/reid.go -p rk3588
```

### Interpreting Results

The above example uses people detected with a YOLOv5 model and then cropped to
create the sample input.

![CK Lady](https://github.com/swdee/go-rknnlite-data/raw/master/reid-walking.jpg)

Objects A1 to A4 represent the same person and objects B1, C1, and D1 are other
people from the same scene.

The first set of comparisons:
```
Comparing object 0 [A1] at (0,0,134,361)
  Object 0 [A1] at (0,0,134,361) has euclidean distance: 0.000000 (same person)
  Object 1 [A2] at (134,0,251,325) has euclidean distance: 0.423271 (same person)
  Object 2 [A3] at (251,0,326,208) has euclidean distance: 0.465061 (same person)
  Object 3 [A4] at (326,0,394,187) has euclidean distance: 0.445583 (same person)
```

Object 0 is A1, when compared to itself it has a euclidean distance of 0.0.  
Objects 1-3 are A2 to A4, each of these have a similar
distance ranging from 0.42 to 0.46.

A euclidean distance range is from 0.0 (same object) to 1.0 (different object), so
the lower the distance the more similar the object is.    A threshold of `0.51` 
is used to define what the maximum distance can be for the object to be considered
the same or different.    Your use case and datasets may require calibration of
the ideal threshold.

The remaining results compare the people B1, C1, and D1.
```
Comparing object 1 [B1] at (394,0,513,357)
  Object 0 [A1] at (0,0,134,361) has euclidean distance: 0.781510 (different person)
  Object 1 [A2] at (134,0,251,325) has euclidean distance: 0.801649 (different person)
  Object 2 [A3] at (251,0,326,208) has euclidean distance: 0.680299 (different person)
  Object 3 [A4] at (326,0,394,187) has euclidean distance: 0.686542 (different person)
Comparing object 2 [C1] at (513,0,588,246)
  Object 0 [A1] at (0,0,134,361) has euclidean distance: 0.860921 (different person)
  Object 1 [A2] at (134,0,251,325) has euclidean distance: 0.873663 (different person)
  Object 2 [A3] at (251,0,326,208) has euclidean distance: 0.870753 (different person)
  Object 3 [A4] at (326,0,394,187) has euclidean distance: 0.820761 (different person)
Comparing object 3 [D1] at (588,0,728,360)
  Object 0 [A1] at (0,0,134,361) has euclidean distance: 0.762738 (different person)
  Object 1 [A2] at (134,0,251,325) has euclidean distance: 0.800668 (different person)
  Object 2 [A3] at (251,0,326,208) has euclidean distance: 0.763694 (different person)
  Object 3 [A4] at (326,0,394,187) has euclidean distance: 0.769597 (different person)
```

All of these other people have a euclidean distance greater than 0.68 indicating
they are different people.


## Postprocessing

[Convenience functions](https://github.com/swdee/go-rknnlite-data/raw/master/postprocess/reid.go) 
are provided for calculating the Euclidean Distance or Cosine Similarity 
depending on how the Model has been trained.