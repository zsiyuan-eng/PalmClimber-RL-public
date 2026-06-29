#include <DynamixelShield.h>
using namespace ControlTableItem;

// ==================== USER CONFIG ====================
const float    DXL_PROTOCOL_VERSION = 1.0;
const uint32_t DXL_BAUDRATE         = 57600;
const uint32_t PC_BAUDRATE          = 115200;

const uint8_t MOTOR_IDS[]  = {1, 2, 3, 4, 5, 6};
const int     MOTOR_COUNT  = 6;

const int BASE_CLIMB_SPEED  = 30;
const int BASE_TURN_SPEED   = 30;
const int SPEED_STEP        = 5;
const int GROUP_OFFSET_STEP = 5;   // applied per press of ',' or '.'

const int MIN_SPEED        = 0;
const int MAX_SPEED        = 200;
const int MIN_GROUP_OFFSET = -100;
const int MAX_GROUP_OFFSET = 100;
// =====================================================

// base "upward" spin direction per motor (hardware-specific, adjust after assembly)
const bool BASE_DIRECTION[] = {
  true,   // ID 1
  false,  // ID 2
  false,  // ID 3
  true,   // ID 4
  true,   // ID 5
  true    // ID 6
};

// turn left: motors 1,4,5 forward / 2,3,6 reverse
const bool TURN_LEFT_DIRECTION[] = {
  true,   // ID 1
  true,   // ID 2
  false,  // ID 3
  false,  // ID 4
  true,   // ID 5
  false   // ID 6
};

// turn right: reverse of left
const bool TURN_RIGHT_DIRECTION[] = {
  false,  // ID 1
  false,  // ID 2
  true,   // ID 3
  true,   // ID 4
  false,  // ID 5
  true    // ID 6
};

const int MOTION_NONE       = 0;
const int MOTION_CLIMB_UP   = 1;
const int MOTION_CLIMB_DOWN = 2;
const int MOTION_TURN_LEFT  = 3;
const int MOTION_TURN_RIGHT = 4;

DynamixelShield dxl;

int  climbSpeed     = BASE_CLIMB_SPEED;
int  turnSpeed      = BASE_TURN_SPEED;
int  frontGroupOffset = 0;   // extra speed bias on motors 1,2,3 (front group)
bool reverseVertical  = false;   // toggled by 'F': flip up/down direction
bool turnMappingFlip  = false;   // toggled by 'R': swap A/D meaning
int  activeMotion     = MOTION_NONE;


int clampValue(int value, int low, int high) {
  if (value < low)  return low;
  if (value > high) return high;
  return value;
}

bool getDirectionForMotor(int motorIndex, int motionType) {
  bool dir = true;
  if (motionType == MOTION_CLIMB_UP) {
    dir = BASE_DIRECTION[motorIndex];
    if (reverseVertical) dir = !dir;
  } else if (motionType == MOTION_CLIMB_DOWN) {
    dir = !BASE_DIRECTION[motorIndex];
    if (reverseVertical) dir = !dir;
  } else if (motionType == MOTION_TURN_LEFT) {
    dir = TURN_LEFT_DIRECTION[motorIndex];
  } else if (motionType == MOTION_TURN_RIGHT) {
    dir = TURN_RIGHT_DIRECTION[motorIndex];
  }
  return dir;
}

int getBaseSpeedForMotion(int motionType) {
  if (motionType == MOTION_CLIMB_UP   || motionType == MOTION_CLIMB_DOWN) return climbSpeed;
  if (motionType == MOTION_TURN_LEFT  || motionType == MOTION_TURN_RIGHT) return turnSpeed;
  return 0;
}

int getMotorSpecificSpeed(int motorIndex, int motionType) {
  int speedMag = getBaseSpeedForMotion(motionType);
  if (motorIndex >= 0 && motorIndex <= 2)   // motors 1/2/3 get the front-group offset
    speedMag += frontGroupOffset;
  return clampValue(speedMag, MIN_SPEED, MAX_SPEED);
}

int speedToGoalValue(bool direction, int speedMag) {
  speedMag = clampValue(speedMag, MIN_SPEED, MAX_SPEED);
  return direction ? speedMag : 1024 + speedMag;
}

void applyMotionToAllMotors(int motionType) {
  for (int i = 0; i < MOTOR_COUNT; i++) {
    bool dir      = getDirectionForMotor(i, motionType);
    int  speedMag = getMotorSpecificSpeed(i, motionType);
    dxl.setGoalVelocity(MOTOR_IDS[i], speedToGoalValue(dir, speedMag));
    delay(5);
  }
}

void stopAllMotors() {
  for (int i = 0; i < MOTOR_COUNT; i++) {
    dxl.setGoalVelocity(MOTOR_IDS[i], 0);
    delay(5);
  }
}

void updateMotionOutput() {
  if (activeMotion == MOTION_NONE) stopAllMotors();
  else                             applyMotionToAllMotors(activeMotion);
}

void initMotor(uint8_t id) {
  dxl.ping(id);
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_VELOCITY);
  dxl.torqueOn(id);
  delay(10);
}

void printStatus() {
  Serial1.print("STATE motion=");    Serial1.print(activeMotion);
  Serial1.print(" climb=");          Serial1.print(climbSpeed);
  Serial1.print(" turn=");           Serial1.print(turnSpeed);
  Serial1.print(" frontOffset=");    Serial1.print(frontGroupOffset);
  Serial1.print(" flipVertical=");   Serial1.print(reverseVertical ? 1 : 0);
  Serial1.print(" flipTurnMap=");    Serial1.println(turnMappingFlip ? 1 : 0);
}

void processCommand(char cmd) {
  switch (cmd) {
    case 'W': activeMotion = MOTION_CLIMB_UP;    break;
    case 'S': activeMotion = MOTION_CLIMB_DOWN;  break;
    case 'A': activeMotion = turnMappingFlip ? MOTION_TURN_RIGHT : MOTION_TURN_LEFT;  break;
    case 'D': activeMotion = turnMappingFlip ? MOTION_TURN_LEFT  : MOTION_TURN_RIGHT; break;
    case '0': activeMotion = MOTION_NONE;        break;
    case 'F': reverseVertical = !reverseVertical; break;
    case 'R': turnMappingFlip = !turnMappingFlip; break;
    case 'I': climbSpeed      = clampValue(climbSpeed + SPEED_STEP, MIN_SPEED, MAX_SPEED); break;
    case 'K': climbSpeed      = clampValue(climbSpeed - SPEED_STEP, MIN_SPEED, MAX_SPEED); break;
    case 'J': turnSpeed       = clampValue(turnSpeed  + SPEED_STEP, MIN_SPEED, MAX_SPEED); break;
    case 'L': turnSpeed       = clampValue(turnSpeed  - SPEED_STEP, MIN_SPEED, MAX_SPEED); break;
    case ',': frontGroupOffset = clampValue(frontGroupOffset + GROUP_OFFSET_STEP, MIN_GROUP_OFFSET, MAX_GROUP_OFFSET); break;
    case '.': frontGroupOffset = clampValue(frontGroupOffset - GROUP_OFFSET_STEP, MIN_GROUP_OFFSET, MAX_GROUP_OFFSET); break;
    default: return;
  }
  updateMotionOutput();
  printStatus();
}

void setup() {
  Serial1.begin(PC_BAUDRATE);
  dxl.begin(DXL_BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  for (int i = 0; i < MOTOR_COUNT; i++) initMotor(MOTOR_IDS[i]);
  stopAllMotors();
  Serial1.println("READY");
  printStatus();
}

void loop() {
  while (Serial1.available() > 0) {
    char cmd = Serial1.read();
    processCommand(cmd);
  }
}
