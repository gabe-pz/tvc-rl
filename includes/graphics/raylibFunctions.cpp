#include "raylibFunctions.h"
#include <raylib.h>

//my (x, y, z) -> raylib (y, z, x), same as cf matrix
Vector3 toRaylib(const std::array<double, 3>& p){
    Vector3 out = (Vector3){(float)p[1], (float)p[2], (float)p[0]};
    return out;
}

//init raylib 
Camera3D raylibInit(int FPS, const std::array<double, 3>& startPosition){

    //*****RAYLIB INITALIZATION***** 
    //viewing screen
    int screenWidth  = 3500;
    int screenHeight = 2000;
    InitWindow(screenWidth, screenHeight, "tvc-model-rocket-sim");

    //raylib syncing things
    SetTargetFPS(FPS); 

    //******CAMERA*****
    Vector3 rocketPos = toRaylib(startPosition);

    Camera3D camera = { 0 };

    //sit camera a fixed offset away from rocket
    camera.position = (Vector3){rocketPos.x + 6.0f, rocketPos.y + 4.0f, rocketPos.z + 6.0f};
    camera.target = rocketPos;
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    DisableCursor();

    return camera;
}

void raylibDrawRocket(const double distanceToThrustVector, const double centerOfGravity, const std::array<double, 4>& stateQ, const std::array<double, 3> position, 
        const std::array<double, 3> lastPos, Camera3D& camera){

    //raylib rocket Dimensions
    float bodyRadius = 0.25f;
    float bodyHeight = 2.5f;
    float coneHeight = 0.7;
    float coneTopRadius = 0.05f; 
    float cgFrac = (distanceToThrustVector - centerOfGravity) / distanceToThrustVector;                             

    //coordinate transformation matrix between raylib vecs and my own vecs. 
    //Form for 4x4 matrix is: r1 u, r2 v, r3 w, r4 t, where u, v, and w is my x, y, and z hat vector mapped into raylibs coordinate system, and t is origin offset
    float cf[16] = {
        0,0,1,0,
        1,0,0,0,
        0,1,0,0,
        0,0,0,1
    };

    //rotation matrix creation
    float rf[16];
    quatToMat(stateQ, rf);

    //*****CAMERA FOLLOW*****
    Vector3 rocketPos = toRaylib(position);
    Vector3 lastRocketPos = toRaylib(lastPos);

    //third person = mouse orbits around target instead of free look
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    //how far the rocket moved since last frame
    Vector3 delta = (Vector3){rocketPos.x - lastRocketPos.x, rocketPos.y - lastRocketPos.y, rocketPos.z - lastRocketPos.z};

    //drag camera along by same amount so it keeps its distance
    camera.position.x = camera.position.x + delta.x;
    camera.position.y = camera.position.y + delta.y;
    camera.position.z = camera.position.z + delta.z;

    camera.target = rocketPos;

    lastRocketPos = rocketPos;

    //*****DRAW ROCKET*****
    BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(camera);

            rlPushMatrix();

                //transform into raylibs coords and move into rocket frame to then get rotation
                rlMultMatrixf(cf);                                   
                rlTranslatef(position[0], position[1], position[2]); 
                rlMultMatrixf(rf);                                   

                //rotate rocket to be facing correct way
                rlRotatef(90.0f, 1.0f, 0.0f, 0.0f);                

                //shift cyclinder down, such that c.g is point rotate about. i.e make it the origin
                rlTranslatef(0.0f, -cgFrac * bodyHeight, 0.0f);
                
                //main rocket body
                DrawCylinder((Vector3){0,0,0}, bodyRadius, bodyRadius, bodyHeight, 64, BLACK);

                //nose cone
                rlPushMatrix();
                    rlTranslatef(0.0f, bodyHeight, 0.0f);
                    DrawCylinder((Vector3){0,0,0}, coneTopRadius, bodyRadius, coneHeight, 64, RED);
                rlPopMatrix();

            rlPopMatrix();

            DrawGrid(100, 1.0f);
        
        EndMode3D();
    EndDrawing();
}

