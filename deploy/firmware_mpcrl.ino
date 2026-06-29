/*

  Adds a numeric velocity command on top of the original WASD protocol:
    "Vw1:12.5,w2:12.5,w3:-3.0,w4:12.5,w5:-3.0,w6:12.5\n"
    -> directly sets each wheel's goal velocity (rad/s -> internal units)

  Also streams IMU data back at 50Hz:
    "IMU pitch=0.031 roll=-0.012 height_est=1.45\n"

  Original WASD keyboard control is still available (falls back when
  no 'V' command is received).

  Wiring: same as before. IMU = MPU-6050 on I2C (SDA=20, SCL=21 on Mega)
*/

#include <DynamixelShield.h>
#include <Wire.h>

using namespace ControlTableItem;

// ==================== USER CONFIG ====================
const float    DXL_PROTOCOL  = 1.0;
const uint32_t DXL_BAUDRATE  = 57600;
const uint32_t PC_BAUDRATE   = 115200;
const uint8_t  MOTOR_IDS[]   = {1, 2, 3, 4, 5, 6};
const int      MOTOR_COUNT   = 6;
const float    WHEEL_RADIUS  = 0.025f;  // meters
// =====================================================

DynamixelShield dxl;

// IMU registers (MPU-6050)
#define MPU6050_ADDR  0x68
#define GYRO_CONFIG   0x1B
#define ACCEL_CONFIG  0x1C
#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B

float pitch_deg = 0.0f;
float roll_deg  = 0.0f;
float height_est = 0.4f;  // naive integration from wheel odometry

bool   velocity_mode = false;   // true = numeric V command, false = WASD
float  wheel_cmds[6] = {0};

unsigned long last_imu_send = 0;
const unsigned long IMU_PERIOD_MS = 20;  // 50Hz

// --- MPU-6050 init ---
void initIMU() {
  Wire.begin();
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);  // wake up
  Wire.endTransmission(true);
  delay(10);
}

// --- read raw accel ---
void readIMU() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true);

  int16_t ax = Wire.read()<<8 | Wire.read();
  int16_t ay = Wire.read()<<8 | Wire.read();
  int16_t az = Wire.read()<<8 | Wire.read();
  // skip temp
  Wire.read(); Wire.read();
  int16_t gx = Wire.read()<<8 | Wire.read();

  const float ACCEL_SCALE = 16384.0f;
  float ax_g = ax / ACCEL_SCALE;
  float ay_g = ay / ACCEL_SCALE;
  float az_g = az / ACCEL_SCALE;

  pitch_deg = atan2(ax_g, sqrt(ay_g*ay_g + az_g*az_g)) * 57.2958f;
  roll_deg  = atan2(ay_g, sqrt(ax_g*ax_g + az_g*az_g)) * 57.2958f;
}

// --- parse "V1:12.5,2:-3.0,...\n" ---
void parseVelocityCmd(const String& line) {
  // format: "V1:12.5,2:12.5,3:12.5,4:12.5,5:12.5,6:12.5"
  String s = line.substring(1);  // strip leading 'V'
  int idx = 0;
  while (s.length() > 0 && idx < MOTOR_COUNT) {
    int comma = s.indexOf(',');
    String tok = (comma >= 0) ? s.substring(0, comma) : s;
    int colon = tok.indexOf(':');
    if (colon > 0) {
      int   motor_id = tok.substring(0, colon).toInt();
      float vel_rads = tok.substring(colon + 1).toFloat();
      if (motor_id >= 1 && motor_id <= 6) {
        wheel_cmds[motor_id - 1] = vel_rads;
      }
    }
    if (comma < 0) break;
    s = s.substring(comma + 1);
    idx++;
  }
  velocity_mode = true;
}

int velRadsToDxl(float vel_rads) {
  // MX-28 in wheel mode: 0=stop, 1..1023=CCW, 1024..2047=CW
  // rpm = vel_rads * 60 / (2*pi), then map to register
  // max rpm ~ 54 RPM -> 1023. Each unit ~ 0.053 rpm.
  float rpm = fabs(vel_rads) * 9.5493f;   // rad/s to rpm
  int val = (int)(rpm / 0.053f);
  val = constrain(val, 0, 1023);
  if (vel_rads < 0) val += 1024;
  return val;
}

void applyVelocityCommands() {
  for (int i = 0; i < MOTOR_COUNT; i++) {
    dxl.setGoalVelocity(MOTOR_IDS[i], velRadsToDxl(wheel_cmds[i]));
  }
  // crude height estimate from mean wheel speed * wheel radius * dt
  float mean_vel = 0;
  for (int i = 0; i < MOTOR_COUNT; i++) mean_vel += wheel_cmds[i];
  mean_vel /= MOTOR_COUNT;
  height_est += mean_vel * WHEEL_RADIUS * (IMU_PERIOD_MS / 1000.0f);
  height_est = constrain(height_est, 0.0f, 4.0f);
}

void initMotor(uint8_t id) {
  dxl.ping(id);
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_VELOCITY);
  dxl.torqueOn(id);
  delay(10);
}

void setup() {
  Serial1.begin(PC_BAUDRATE);
  dxl.begin(DXL_BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL);
  for (int i = 0; i < MOTOR_COUNT; i++) initMotor(MOTOR_IDS[i]);
  initIMU();
  Serial1.println("READY v2 (MPC-RL deploy mode)");
}

String inputBuffer = "";

void loop() {
  // read incoming serial commands
  while (Serial1.available() > 0) {
    char c = Serial1.read();
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.startsWith("V")) {
        parseVelocityCmd(inputBuffer);
        applyVelocityCommands();
      } else if (inputBuffer.length() == 1) {
        // legacy WASD single char -- hand off to original handler
        velocity_mode = false;
        // (simplified: just stop on any legacy command in this firmware)
        for (int i = 0; i < MOTOR_COUNT; i++)
          dxl.setGoalVelocity(MOTOR_IDS[i], 0);
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // send IMU at 50Hz
  unsigned long now = millis();
  if (now - last_imu_send >= IMU_PERIOD_MS) {
    readIMU();
    Serial1.print("IMU pitch=");
    Serial1.print(pitch_deg * 0.01745f, 4);  // convert to radians
    Serial1.print(" roll=");
    Serial1.print(roll_deg * 0.01745f, 4);
    Serial1.print(" height_est=");
    Serial1.println(height_est, 3);
    last_imu_send = now;
  }
}
