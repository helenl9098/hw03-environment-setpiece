#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

// based off of http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
float intersectSDF(float distA, float distB) {
    return max(distA, distB);
}

float unionSDF(float distA, float distB) {
    return min(distA, distB);
}

float differenceSDF(float distA, float distB) {
    return max(distA, -distB);
}

// based off of http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); 
}

float opSmoothIntersection( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h); 
}


mat4 rotateX(float theta) {
    float c = cos(radians(theta));
    float s = sin(radians(theta));

    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, c, -s, 0),
        vec4(0, s, c, 0),
        vec4(0, 0, 0, 1)
    );
}

mat4 rotateZ(float theta) {
    float c = cos(radians(theta));
    float s = sin(radians(theta));

    return mat4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    );
}

float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

float sdCappedCylinder( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}


float sdCappedCone( vec3 p, float h, float r1, float r2 )
{
    vec2 q = vec2(length(p.xz), p.y );
    
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2, k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot(ca, ca),dot(cb, cb)) );
}

float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float myRandomMagic( vec3 p , vec3 seed) {
  return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

// ----------------- END OF HELPER FUNCTIONS -------------- //

struct Intersection {
  vec3 point;
  int objectID;
};

Intersection currentInt;

// lantern displacement
float displace(vec3 p) {
  return abs(sin(p.y * 4.0)) * 0.04;
}

// lantern displacement 2
float displace2(vec3 p) {
  return abs(sin(p.x * 2.5) * sin(p.z * 2.5)) * 0.02;
}

// roof displacement
float displaceRoof(vec3 p) {
  return abs(sin(p.z * 1.0)) * 0.2;
}

// lantern sdf
float lantern(vec3 p) {
  float s = sdSphere(p, 5.0);
  float displacement = displace2(p) + displace(p);
  s += displacement;

  float top = sdCappedCylinder(p - vec3(0, 4.5, 0), vec2(2.5, 0.5));
  if (top < 0.00001) {
    currentInt = Intersection(p, 4);
    return 0.0;
  }
  float string = sdCappedCylinder(p - vec3(0, 4.5, 0), vec2(0.05, 5.0));
  if (string < 0.00001) {
    currentInt = Intersection(p, 5);
    return 0.0;
  }
  top = unionSDF(top, string);
  s = unionSDF(top, s);
  return s;
}

float sceneWithoutLanterns(vec3 p) {
  vec3 point = p;

  vec3 roofPoint = vec3(vec4(point, 1.0) * rotateZ(-15.0));

  float roof = sdBox(roofPoint - vec3(12.0, 9.0, 0.0), vec3(11, 1.0, 50));
  roof -= displaceRoof(point);

  float roof2 = sdBox(roofPoint - vec3(12.0, 7.5, 0.0), vec3(12, 1.0, 50));
  roof2 -= displaceRoof(point);

  roof = differenceSDF(roof, roof2);

  if (roof < 0.00001) {
    currentInt = Intersection(p, 1);
    return 0.0;
  }

  float wall = sdBox(point - vec3(20.0, 10.0, 0.0), vec3(0.4, 50, 50));
  if (wall < 0.01) {
    currentInt = Intersection(p, 2);
    return 0.0;
  }

  float pillar = sdCappedCylinder(point - vec3(12.0, 0.0, 41.0), vec2(3.0, 20.0));
  float pillar2 = sdCappedCylinder(point - vec3(12.0, 0.0, 7.0), vec2(3.0, 20.0));
  float pillars = unionSDF(pillar, pillar2);

  if (pillars < 0.00001) {
    currentInt = Intersection(p, 3);
    return 0.0;
  }

  float entireScene = unionSDF(pillars, roof);
  entireScene = unionSDF(entireScene, wall);
  return entireScene;
}

// whole scene
float sceneSDF(vec3 p) {
  float lantern0 = lantern(p - vec3(7.0, 2.0, 16.0));
  float lantern1 = lantern(p - vec3(7.8, 2.5, -2.0));
  float lantern2 = lantern(p - vec3(7.5, 3.5, 34.0));

  lantern1 = unionSDF(lantern1, lantern0);
  lantern1 = unionSDF(lantern1, lantern2);

  float restOfScene = sceneWithoutLanterns(p);
  float entire = unionSDF(restOfScene, lantern1);

  return entire;
}

vec3 estimateNormal(vec3 p) {
  float EPSILON = 0.001;
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

vec3 getRayDirection() {

  float fovy = 30.0;
  vec3 look = normalize(u_Ref - u_Eye);
  vec3 right = normalize(cross(look, u_Up));
  vec3 up = cross(right, look);

  float tan_fovy = tan(radians(fovy / 2.0));
  float len = length(u_Ref - u_Eye);
  float aspect = u_Dimensions.x / float(u_Dimensions.y);

  vec3 v = up * len * tan_fovy;
  vec3 h = right * len * aspect * tan_fovy;

  vec3 p = u_Ref + fs_Pos.x * h + fs_Pos.y * v;
  vec3 dir = normalize(p - u_Eye);

  return dir;

}


// soft shadows whole scene
// based off of http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    for( float t=mint; t < maxt; )
    {
        float h = sceneSDF(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

// soft shadows - lanterns
// based off of http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softshadowLantern1( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    for( float t=mint; t < maxt; )
    {
        vec3 p = ro + rd*t;
        float h = sceneWithoutLanterns(p);
        if( h<0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}


// returns 3D value noise and its 3 derivatives
vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    vec3 seed = vec3(80, 0, 0);

    float a = myRandomMagic( p+vec3(0,0,0), seed );
    float b = myRandomMagic( p+vec3(1,0,0), seed );
    float c = myRandomMagic( p+vec3(0,1,0), seed );
    float d = myRandomMagic( p+vec3(1,1,0), seed );
    float e = myRandomMagic( p+vec3(0,0,1), seed );
    float f = myRandomMagic( p+vec3(1,0,1), seed );
    float g = myRandomMagic( p+vec3(0,1,1), seed );
    float h = myRandomMagic( p+vec3(1,1,1), seed );

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), 
                      2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                                      k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                                      k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

// fbm noise
float fbm(vec3 p) {
    p /= 1.0;
  // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    
  // Loop of octaves
    for (int i = 0; i < 10; i++) {
        value += amplitude * noised(p).y;
        p *= 2.;
        amplitude *= .5;
    }

    return value / 0.5 + 0.7;
}

// based off of : https://www.shadertoy.com/view/MscXD7
#define _SnowflakeAmount 300  // Number of snowflakes
#define _BlizardFactor 0.2    // Fury of the storm !

vec2 uv;

float rnd(float x) {
    return fract(sin(dot(vec2(x+47.49,38.2467/(x+2.3)), vec2(12.9898, 78.233)))* (43758.5453));
}

float drawCircle(vec2 center, float radius) {
    return 1.0 - smoothstep(0.0, radius, length(uv - center));
}

// returns color
vec3 rayMarch(vec3 dir) {

  // LIGHT 1
  vec3 light1Point = vec3(-10.0, 16.0, -2.0);

  // Lantern Lights
  vec3 lantern1Light = vec3(-8.0, 0.0, -2.0);
  vec3 lantern1Light2 = vec3(1.0, 0.0, -2.0);
  vec3 lantern1Light3 = vec3(1.0, 0.0, 13.0);
  vec3 lantern1Light4 = vec3(1.0, 0.0, 34.0);

  // LIGHT 2
  vec3 light2Point = vec3(0.0, -40.0, 10.0);


  // ray marching constants
  float depth = 0.0; 
  int MAX_MARCHING_STEPS = 1000;
  float EPSILON = 0.00001;
  float MAX_TRACE_DISTANCE = 500.0;

  // snow!!
  // based off of https://www.shadertoy.com/view/MscXD7
  vec4 fragCoord = gl_FragCoord;
  uv = fragCoord.xy / u_Dimensions.x;
  vec4 fragColor = vec4(0.0, 0.0, 0.0, 1.0);
  float j;
  float time = u_Time * 0.003;
  for(int i=0; i<_SnowflakeAmount; i++)
  {
        j = float(i);
        float speed = 0.1+rnd(cos(j))*(0.7+0.5*cos(j/(float(_SnowflakeAmount)*0.25)));
        vec2 center = vec2((0.25-uv.y)*_BlizardFactor+rnd(j)+0.01*cos(time+sin(j)), mod(sin(j)-speed*(time*9.5*(0.1+_BlizardFactor)), 0.65));
        fragColor += vec4(0.5*drawCircle(center, 0.001+speed*0.013));
  }
  fragColor *= vec4(1.0, 0.8, 0.8, 1.0);

    // vignette
    vec2 centerCoords = vec2(u_Dimensions.x / 2.0,
                         u_Dimensions.y / 2.0);
    float maxDistance = sqrt(pow(centerCoords.x, 2.0) +
                             pow(centerCoords.y, 2.0));
    float shortX = fragCoord.x - centerCoords.x;
    float shortY = fragCoord.y - centerCoords.y;
    float currentDistance = pow(shortX, 2.0) / pow(u_Dimensions.x, 2.0) +
                            pow(shortY, 2.0) / pow(u_Dimensions.y, 2.0);

    float intensity = 2.3; // how intense the vignette is
    float vignette = currentDistance * intensity;
    float intensity2 = 5.3; // how intense the vignette is
    float vignette2 = currentDistance * intensity2;
    vec3 vignetteColor = mix(vec3(1.0), vec3(0, 0, 0.6), vignette);
    vec3 vignetteColor2 = mix(vec3(0.0), vec3(1.0, 0.9, 0.7), (1.0 - vignette2));



    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
      vec3 point = u_Eye + depth * dir;

      float dist = sceneSDF(point);

      if (dist < EPSILON) {

        // distance fog
        float dist = point.z;
        const vec3 fogColor = vec3(0.0, 0.0,0.1);
        float fogFactor = 0.;
        fogFactor = (70. - dist)/(70.0 - 30.0);
        fogFactor = clamp( fogFactor, 0.0, 1.0 );

        vec3 normal = normalize(vec3(-1, -1, -1) * estimateNormal(point));

        // light shadows
        float lcolor1 = softshadowLantern1(point, 
                                normalize(lantern1Light2 - point), 
                                2.0, 100.0, 2.0);

        float lcolor12 = softshadowLantern1(point, 
                                normalize(lantern1Light3 - point), 
                                2.0, 100.0, 2.0);
        float lcolor13 = softshadowLantern1(point, 
                                normalize(lantern1Light4 - point), 
                                2.0, 100.0, 2.0);

        float color1 = softshadow(point, 
                                normalize(light1Point - point), 
                                2.0, 100.0, 2.0);
        float color2 = softshadow(point, 
                                normalize(light2Point - point), 
                                2.0, 100.0, 2.0) + 0.4;


        // lambert diffuse terms
        vec3 diffuseColor = vec3(1.0, 0.0, 0.0);
        float diffuseTerm = dot(normalize(vec3(-1, -1, -1) * estimateNormal(point)), normalize(vec3(1.0, 3.0, -7.0) - u_Eye));
        float specularIntensity = max(pow(dot(normalize(vec3(1.0, 3.0, -7.0) - u_Eye), normalize(point - vec3(1.0, 3.0, -7.0))), 80.0), 0.0);
        diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
        float ambientTerm = 0.2;
        float lightIntensity = diffuseTerm + ambientTerm; 

        // lantern color
        if (currentInt.objectID == 0) {
          
          diffuseColor = vec3(0.9, 0.0, 0.0) * lightIntensity;
          diffuseColor += vec3(0.9, 0.15, 0.15) * lightIntensity * lightIntensity * lightIntensity;

          diffuseColor += vec3(fbm(point)) * 0.06;
          diffuseColor *= vec3(1.0 - abs(normal.y)) + 0.2;
          float randomWall = myRandomMagic(point * 0.1, vec3(10, 0, 0)) * 0.13;
          diffuseColor += vec3(randomWall);
        }

        // roof color
        if (currentInt.objectID == 1) {
          float randomWall = myRandomMagic(point * 0.1, vec3(10, 0, 0)) * fbm(point * 15.0) * 0.08;
          diffuseColor = vec3(0.9, 0.1, 0.1) * lightIntensity + randomWall;
        }

        // wall color
        if (currentInt.objectID == 2) {
          vec3 wallColorA = vec3(180, 150, 150) / 255.;
          vec3 wallColorB = vec3(150, 100, 150) / 255.;
          float randomWall = myRandomMagic(point * 0.1, vec3(10, 0, 0)) * fbm(point * 0.2);
          vec3 wallColor = mix(wallColorA, wallColorB, randomWall);

          vec3 shadowA = vec3(0.0, 0.0, 0.4);
          vec3 shadowB = vec3(0.8, 0.4, 0.0);
          vec3 shadow = mix(shadowA, shadowB, (1.0 - color1));
          diffuseColor = wallColor * shadow;
        }

        // wooden pillar color
        if (currentInt.objectID == 3) {
          vec3 woodPoint = vec3(point.x * 2.0, point.y * 0.2, point.z);
          vec3 woodColorA = vec3(63, 38, 30) / 255.;
          vec3 woodColorB = vec3(30, 11, 10) / 255.;
          float randomWall = myRandomMagic(point * 0.1, vec3(10, 0, 0)) * fbm(point) * 0.05;
          diffuseColor = mix(woodColorA, woodColorB, smoothstep(0.0, 1.0, fbm(woodPoint))) + randomWall;
       
        }

        // lantern rim color
        if (currentInt.objectID == 4) {
          diffuseTerm = dot(normalize(vec3(-1, -1, -1) * estimateNormal(point)), normalize(point - u_Eye));
          specularIntensity = max(pow(dot(normalize(point - u_Eye), normalize(normal)), 40.0), 0.0);

        
          float PI = 3.14159265358979323846;
  
          vec3 color = vec3(0.6 + 0.5 * cos(2. * PI * (1.0 * diffuseTerm + 0.00)),
                            0.6 + 0.5 * cos(2. * PI * (0.6 * diffuseTerm + 0.19)),
                            0.6 + 0.5 * cos(2. * PI * (0.4 * diffuseTerm + 0.22)));
          diffuseColor = color + vec3(specularIntensity) * vec3(1.0, 0.0, 0.0);
        }

        // string color
        if (currentInt.objectID == 5) {
          diffuseColor = vec3(0.2) * lightIntensity;
        }

        // lantern lights
        vec3 colorC = vec3(0.0, 0.0, 30) / 255.;
        vec3 colorD = vec3(255, 214, 150) / 255.;

        float allLights = (lcolor1 + lcolor12 + lcolor13) / 3.0;
        vec3 orangeBlue = mix(colorC, colorD, allLights + 0.2);
        
        vec3 finalColor = diffuseColor * orangeBlue;
        finalColor = mix(fogColor, finalColor, fogFactor) 
                     * mix(vec3(1.0), vec3(246, 0, 255) / 255., pow(color1, 0.5))
                     * mix(vec3(1.0), vec3(106, 247, 255) / 255., pow(color2, 0.5));



        // snow
        if (fragCoord.x < u_Dimensions.x / 2.0 + 15.0) {
          return finalColor * vignetteColor;
        }
        else {
          return (finalColor + (vec3(fragColor) * vignetteColor2)) * vignetteColor;
        }
        
      }

      // keep going!
      depth += dist;

      // we went too far ... we should stop
      if (depth >= MAX_TRACE_DISTANCE) {
           
        // snow background
        return vec3(fragColor) * 0.3 * vignetteColor2;
      }
    }
    return vec3(0, 0, 0);
}


void main() {

  vec3 dir = getRayDirection();
  out_Col = vec4(rayMarch(dir), 1.0);
}
