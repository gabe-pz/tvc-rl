#include "rwParameters.h"



void writeParameters(const std::array<std::array<float, 4>, 100>& w1, const std::array<std::array<float, 100>, 100>& w2, const std::array<std::array<float, 100>, 4>& w3, const std::array<float, 100>& b1, 
        const std::array<float, 100>& b2, const std::array<float, 4>& b3){

    std::ofstream file("parameters.bin", std::ios::binary);

    //write w1
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 4; j++){
            float val = w1[i][j];
            file.write((const char*)&(val), sizeof(float));
        }
    }

    //write w2
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            float val = w2[i][j];
            file.write((const char*)&(val), sizeof(float));
        }
    }

    //write w3
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 100; j++){
            float val = w3[i][j];
            file.write((const char*)&(val), sizeof(float));
        }
    }


    //write b1, b2, b3
    for(int i = 0; i < 100; i++){
        file.write((const char*)&(b1[i]), sizeof(float));
        file.write((const char*)&(b2[i]), sizeof(float));
    }

    for(int i = 0; i < 4; i++){
        file.write((const char*)&(b3[i]), sizeof(float));
    }

    std::cout << "\n****************" << std::endl;
    std::cout << "WRITE OF PARAMETERS COMPLETE!" << std::endl;
}


void loadParameters(std::array<std::array<float, 4>, 100>& w1, std::array<std::array<float, 100>, 100>& w2, std::array<std::array<float, 100>, 4>& w3, std::array<float, 100>& b1, 
        std::array<float, 100>& b2, std::array<float, 4>& b3){

    std::ifstream file("parameters.bin", std::ios::binary);


    //read w1
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 4; j++){
            float tempVal = 0.0f;
            file.read((char*)&(tempVal), sizeof(float));//take the bytes from .bin, copy them to tempVal addr 
            w1[i][j] = tempVal;//set the value for current w1 element to whats at that tempVal addr
        }
    }

    //read w2
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            float tempVal = 0.0f;
            file.read((char*)&(tempVal), sizeof(float));//take the bytes from .bin, copy them to tempVal addr 
            w2[i][j] = tempVal;//set the value for current w1 element to whats at that tempVal addr
        }
    }

    //read w3
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 100; j++){
            float tempVal = 0.0f;
            file.read((char*)&(tempVal), sizeof(float));//take the bytes from .bin, copy them to tempVal addr 
            w3[i][j] = tempVal;//set the value for current w1 element to whats at that tempVal addr
        }
    }

    //read b1, b2, and b3
    for(int i = 0; i < 100; i++){
        float tempVal = 0.0f;

        file.read((char*)&(tempVal), sizeof(float));
        b1[i] = tempVal;

        file.read((char*)&(tempVal), sizeof(float));
        b2[i] = tempVal;
    }
    for(int i = 0; i < 4; i++){
        float tempVal = 0.0f;
        file.read((char*)&(tempVal), sizeof(float));
        b3[i] = tempVal;
    }
    
    std::cout << "\n****************" << std::endl;
    std::cout << "LOADED PARAMETERS!" << std::endl;
}




