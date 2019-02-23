# Lanterns - Hanyu Liu (liuhanyu)

## Inspiration
![](nigel-tadyanehondo-199310-unsplash.jpg)

## Live Demo Link 

https://helenl9098.github.io/hw03-environment-setpiece/

## Implementation Details

- __Lanterns__: Made by SDFs displaced in the Y direction (and subtlely in the x and z direction) using sin functions. The positions are offset by noise values. Lantern body textured using FBM noise and overlayed speckled noise. Lantern rim textured using cosine color palette with specular lighting.

- __Snow Animation__: Animated by generating circles of different sizes determined by noise over a period of time

- __Walls__: Made by SDFs and textured using FBM noise and speckled noise.

- __Roof__: Made by SDFs displaced in the Z direction and textured using FBM noise and overlayed speckled noise.

- __Wooden Pillars__: Made by SDFs and textured using FBM noise displaced in the Y direction and overlayed speckled/FBM noise.

- __Lighting__: There are SDF-based soft shadows and ambient lighting originating from each of the three lanterns. There are also 2 directional light sources coming subtlely from the bottom and top, giving the lanterns a green shade from the bottom and purple shade from the top. All materials are diffuse / lambert lighting except the lantern rims which are specular (Blinn-Phong) lighting. The shadows / lights are also mapped from [0, 1] to colors. For example, the shadows are dark purple and the lights of the lanterns are orange. 

- __Other Effects__: Vingette effect around the edges. Distance fog that fades the geometry to dark purple. 


## External Resources

- https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
- http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
- https://www.shadertoy.com/view/MscXD7
- http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
- https://unsplash.com/photos/btyl4ggZJv4

