#include "rocketPhysics.h"

void stateUpdate(const double dt, double t, double U, double sigmaU, float actionX, float actionY, float servoXOffset, float servoYOffset, 
        std::array<std::vector<double>, 3> pinkNoise, std::array<double, 3>& position, std::array<double, 3>& velocity, std::array<double, 4>& stateQ, 
        std::array<double, 4>& stateQMeasure, std::array<double, 3>& angularAccleration, std::array<double, 3>& angularVelocity, std::array<double, 2>& psi, 
        std::array<double, 3>& angularVelocityMeasure, std::array<double, 2>& psiMeasure, std::mt19937& gen){ 
    //*****CONSTANT SCALARS***** 
    //rocket constants
    constexpr double centerOfPressure = 0.0877;
    constexpr double aRef = 0.00456; 
    constexpr double cD = 0.291;
    constexpr int cNa = 2;  

    constexpr double Ixx = 0.02498995;//**to-do: change to vary with mass
    constexpr double Iyy = 0.02498688;

    //world constants
    constexpr double gravity = 9.81;  
    constexpr float rho = 1.187f;

    constexpr double angleWindHeadingX = 0.0; 
    constexpr double angleWindHeadingY = 0.0; 

    //wind velocity in x and y 
    static double ux = std::cos(angleWindHeadingX);
    static double uy = std::sin(angleWindHeadingY); 


    //*****CONSTANT VECTORS*****
    constexpr std::array<double, 3> r = {0.0, 0.0, centerOfGravity-distanceToThrustVector};
    constexpr std::array<double, 3> rAero = {0.0, 0.0, centerOfGravity-centerOfPressure};


    
    //*****WIND VELOCITY*****
    //sampling three independent turbulence streams at time t
    double u = windVelocity(t, U, sigmaU, pinkNoise[0]);   
    double v = windVelocity(t, 0.0, 0.8*sigmaU, pinkNoise[1]);   
    double w = windVelocity(t, 0.0, 0.5*sigmaU, pinkNoise[2]);  

    //rotate from wind frame to world frame
    std::array<double, 3> windVelocityWf = {u*ux - v*uy, u*uy + v*ux, w};



    //*****RELATIVE VELOCITY***** 
    //velocity of rocket wrt to wind in world frame, then into rocket frame
    std::array<double, 3> relativeVelocityWf = {velocity[0] - windVelocityWf[0], velocity[1] - windVelocityWf[1], velocity[2] - windVelocityWf[2]};
    std::array<double, 3> relativeVelocityRf = rotateWfToRf(stateQ, relativeVelocityWf);
    
    //magnitude of relative v in RF
    double relativeVelMag = std::sqrt((relativeVelocityRf[0]*relativeVelocityRf[0]) + (relativeVelocityRf[1]*relativeVelocityRf[1]) + (relativeVelocityRf[2]*relativeVelocityRf[2]));



    //*****SUM OF FORCES*****
    //force due to thrust
    std::array<double, 3> thrustRf = forceThrustRf(actionX+servoXOffset, actionY+servoYOffset, t);
    std::array<double, 3> thrustWf = rotateRfToWf(stateQ, thrustRf); 

    //aero forces. Note for normal force throw the negative on there to account for the fact want to use the free-stream velocity
    std::array<double, 3> aerodynamicForcesRf = {-0.5*rho*cNa*aRef*relativeVelocityRf[0]*relativeVelMag, -0.5*rho*cNa*aRef*relativeVelocityRf[1]*relativeVelMag, 
                        -0.5*rho*cD*aRef*(std::abs(relativeVelocityRf[2]))*relativeVelocityRf[2]};
    
    std::array<double, 3> aerodynamicForceswf = rotateRfToWf(stateQ, aerodynamicForcesRf);

    //sum forces
    std::array<double, 3> sumOfForcesWf = {thrustWf[0]+aerodynamicForceswf[0], thrustWf[1]+aerodynamicForceswf[1], thrustWf[2]-mass(t)*gravity+aerodynamicForceswf[2]}; 
    

    
    //****POSITION*****
    //compute accleration
    std::array<double, 3> accleration = {sumOfForcesWf[0] / mass(t), sumOfForcesWf[1] / mass(t), sumOfForcesWf[2] / mass(t)};//F=dp/dt

    //integrate accleration for velocity
    velocity[0] += dt*accleration[0];
    velocity[1] += dt*accleration[1];
    velocity[2] += dt*accleration[2];

    //integrate velocity for position 
    position[0] += dt*velocity[0];
    position[1] += dt*velocity[1];
    position[2] += dt*velocity[2]; 



    //*****SUM OF TORQUES*****
    std::array<double, 3> torqueThrust = crossProduct(r, thrustRf);
    std::array<double, 3> torqueAero = crossProduct(rAero, aerodynamicForcesRf);
    std::array<double, 3> sumOfTorques = {torqueThrust[0]+torqueAero[0], torqueThrust[1]+torqueAero[1], torqueThrust[2]+torqueAero[2]};

    
    //****ANGULAR POSITION***** 
    //compute angular accleration
    angularAccleration = {sumOfTorques[0]/ Ixx, sumOfTorques[1]/ Iyy};//Tau=I(alpha)

    //integrate angular accleration for angular velocity 
    angularVelocity[0] += dt*angularAccleration[0];
    angularVelocity[1] += dt*angularAccleration[1];

    //determine the measured angular velocity to simulate sensor readings                     
    std::normal_distribution<double> gaussianDist(0, deg2rad(0.005)/std::sqrt(dt)); 
    angularVelocityMeasure[0] = angularVelocity[0] + gaussianDist(gen);
    angularVelocityMeasure[1] = angularVelocity[1] + gaussianDist(gen);

    //convert angular velocity to pure quaternion
    std::array<double, 4> angularVelocityQ = vectorToPureQuaternion(angularVelocity);
    std::array<double, 4> angularVelocityMeasureQ = vectorToPureQuaternion(angularVelocityMeasure);

    //compute first derivative of quaternion
    std::array<double, 4> stateQTimeDerivative = multiplyQP(stateQ, angularVelocityQ);
    stateQTimeDerivative[0] = stateQTimeDerivative[0]*0.5;
    stateQTimeDerivative[1] = stateQTimeDerivative[1]*0.5;
    stateQTimeDerivative[2] = stateQTimeDerivative[2]*0.5;
    stateQTimeDerivative[3] = stateQTimeDerivative[3]*0.5;

    //compute first derivative of quaternion
    std::array<double, 4> stateQTimeDerivativeMeasure = multiplyQP(stateQMeasure, angularVelocityMeasureQ);
    stateQTimeDerivativeMeasure[0] = stateQTimeDerivativeMeasure[0]*0.5;
    stateQTimeDerivativeMeasure[1] = stateQTimeDerivativeMeasure[1]*0.5;
    stateQTimeDerivativeMeasure[2] = stateQTimeDerivativeMeasure[2]*0.5;
    stateQTimeDerivativeMeasure[3] = stateQTimeDerivativeMeasure[3]*0.5;


    //integrate time derivative of q to update state quaternion
    stateQ[0] += dt*stateQTimeDerivative[0];
    stateQ[1] += dt*stateQTimeDerivative[1];
    stateQ[2] += dt*stateQTimeDerivative[2];
    stateQ[3] += dt*stateQTimeDerivative[3];

    stateQMeasure[0] += dt*stateQTimeDerivativeMeasure[0];
    stateQMeasure[1] += dt*stateQTimeDerivativeMeasure[1];
    stateQMeasure[2] += dt*stateQTimeDerivativeMeasure[2];
    stateQMeasure[3] += dt*stateQTimeDerivativeMeasure[3];

    //normalize quaterinon 
    normalizeQuaternion(stateQ);
    normalizeQuaternion(stateQMeasure);

    //conver to euler angles
    psi = quaternionToEuler(stateQ); 

    psiMeasure = quaternionToEuler(stateQMeasure); 
}
