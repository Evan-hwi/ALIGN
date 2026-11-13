// imu_angle
// 2차 실습 1~2단계: 자이로 바이어스 보정 + 상보 필터로 센서별 roll/pitch 계산
//
// 배선: ESP32-WROOM-32, SDA = GPIO 21, SCL = GPIO 22
//       IMU 1 (0x68, AD0 -> GND) = 흉추, IMU 2 (0x69, AD0 -> 3V3) = 요추
// 설정: 100Hz, DLPF 약 44Hz, 자이로 ±250°/s, 가속도 ±2g, I2C 400kHz
//
// 주의: roll/pitch는 센서 좌표계 기준이다. 조끼에 붙였을 때 어느 각도가
//       몸의 앞뒤 굽힘인지는 3단계(장착 위치·좌표계 정의)에서 정한다.
//       pitch가 ±90°에 가까우면 roll 값이 불안정해진다(짐벌락).

#include <Wire.h>
#include <math.h>

#define MPU1_ADDR 0x68     // 흉추
#define MPU2_ADDR 0x69     // 요추
#define SAMPLE_US 10000    // 10ms = 100Hz
#define ALPHA 0.98f        // 자이로 신뢰 비율
#define MAX_DT 0.05f       // 이보다 오래 끊기면 자이로 적분 대신 가속도 각도로 다시 맞춤

// 바이어스 측정 시 "정지 상태" 판정 기준
#define CAL_SAMPLES 1000
#define CAL_MAX_GYRO_STD 0.5f    // °/s, 축별 표준편차
#define CAL_MAX_ACC_ERR 0.10f    // g, 가속도 크기와 1g의 차이
#define CAL_RETRIES 3

struct Imu {
  uint8_t addr;
  float gBiasX, gBiasY, gBiasZ;   // 자이로 바이어스 (°/s)
  float roll, pitch;              // 상보 필터 결과 각도 (°)
  float accRoll, accPitch;        // 가속도만으로 구한 각도 (비교용)
  bool ok;                        // 초기화·보정 성공 여부
  uint32_t lastUs;                // 마지막으로 각도를 갱신한 시각 (dt 계산용)
};

Imu imu1 = {MPU1_ADDR};
Imu imu2 = {MPU2_ADDR};

bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool readReg(uint8_t addr, uint8_t reg, uint8_t &val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return false;
  val = Wire.read();
  return true;
}

bool initMPU(uint8_t addr) {
  if (!writeReg(addr, 0x6B, 0x00)) return false;   // 슬립 해제
  delay(100);
  bool ok = writeReg(addr, 0x1A, 0x03)    // DLPF 약 44Hz (내부 1kHz)
         && writeReg(addr, 0x19, 9)       // 1kHz / (9+1) = 100Hz
         && writeReg(addr, 0x1B, 0x00)    // 자이로 ±250°/s
         && writeReg(addr, 0x1C, 0x00);   // 가속도 ±2g
  if (!ok) return false;

  uint8_t id = 0;
  if (!readReg(addr, 0x75, id)) return false;
  if (id != 0x68) {
    // 호환 칩은 다른 값이 나올 수 있어 경고만 하고 진행
    Serial.printf("WARN 0x%02X: WHO_AM_I = 0x%02X (0x68 아님)\n", addr, id);
  }
  return true;
}

bool readMPU(uint8_t addr, float &ax, float &ay, float &az,
             float &gx, float &gy, float &gz) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)14) != 14) return false;

  int16_t rAX = (Wire.read() << 8) | Wire.read();
  int16_t rAY = (Wire.read() << 8) | Wire.read();
  int16_t rAZ = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();   // 온도
  int16_t rGX = (Wire.read() << 8) | Wire.read();
  int16_t rGY = (Wire.read() << 8) | Wire.read();
  int16_t rGZ = (Wire.read() << 8) | Wire.read();

  ax = rAX / 16384.0f;  ay = rAY / 16384.0f;  az = rAZ / 16384.0f;
  gx = rGX / 131.0f;    gy = rGY / 131.0f;    gz = rGZ / 131.0f;
  return true;
}

// 정지 상태에서 평균을 구해 바이어스로 저장한다.
// 측정 중 센서가 움직이면(표준편차·가속도 크기 기준) 버리고 다시 측정한다.
bool calibrateGyro(Imu &m) {
  for (int attempt = 1; attempt <= CAL_RETRIES; attempt++) {
    double s[3] = {0, 0, 0}, s2[3] = {0, 0, 0}, accNorm = 0;
    int n = 0;
    for (int i = 0; i < CAL_SAMPLES; i++) {
      float ax, ay, az, g[3];
      if (readMPU(m.addr, ax, ay, az, g[0], g[1], g[2])) {
        for (int k = 0; k < 3; k++) { s[k] += g[k]; s2[k] += (double)g[k] * g[k]; }
        accNorm += sqrt(ax * ax + ay * ay + az * az);
        n++;
      }
      delay(2);
    }
    if (n < CAL_SAMPLES / 2) {
      Serial.printf("0x%02X 보정 실패: 읽기 성공 %d/%d\n", m.addr, n, CAL_SAMPLES);
      return false;
    }

    float mean[3], stdMax = 0;
    for (int k = 0; k < 3; k++) {
      mean[k] = s[k] / n;
      float var = s2[k] / n - (double)mean[k] * mean[k];
      stdMax = fmaxf(stdMax, sqrtf(fmaxf(var, 0)));
    }
    float accErr = fabsf(accNorm / n - 1.0f);

    if (stdMax <= CAL_MAX_GYRO_STD && accErr <= CAL_MAX_ACC_ERR) {
      m.gBiasX = mean[0];  m.gBiasY = mean[1];  m.gBiasZ = mean[2];
      Serial.printf("0x%02X bias: %.3f, %.3f, %.3f (std %.2f, |a|-1g %.3f)\n",
                    m.addr, m.gBiasX, m.gBiasY, m.gBiasZ, stdMax, accErr);
      return true;
    }
    Serial.printf("0x%02X 움직임 감지 (std %.2f °/s, |a|-1g %.3f g) - 다시 측정 %d/%d\n",
                  m.addr, stdMax, accErr, attempt, CAL_RETRIES);
  }
  Serial.printf("0x%02X 보정 실패: 센서를 고정하고 리셋하세요\n", m.addr);
  return false;
}

// 가속도(중력 방향)로 기울기 계산
void accelAngles(float ax, float ay, float az, float &roll, float &pitch) {
  roll  = atan2f(ay, az) * 180.0f / PI;
  pitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
}

// -180 ~ 180 범위로 맞춤
float wrap180(float a) {
  while (a > 180.0f) a -= 360.0f;
  while (a < -180.0f) a += 360.0f;
  return a;
}

// 상보 필터: 가속도 각도를 기준으로, 자이로로 예측한 각도와의 차이를 ALPHA만큼 반영.
// 차이를 wrap180 하므로 roll이 ±180°를 넘나들어도 평균이 엉뚱하게 0°로 튀지 않는다.
float complementary(float prev, float rate, float accAngle, float dt) {
  float predicted = prev + rate * dt;
  return wrap180(accAngle + ALPHA * wrap180(predicted - accAngle));
}

// 센서 하나 업데이트: 읽기 → 바이어스 제거 → 상보 필터
// dt는 고정 0.01초가 아니라 이 센서를 마지막으로 갱신한 뒤 실제로 흐른 시간
bool updateImu(Imu &m, uint32_t now) {
  float ax, ay, az, gx, gy, gz;
  if (!readMPU(m.addr, ax, ay, az, gx, gy, gz)) return false;
  float dt = (now - m.lastUs) / 1e6f;
  m.lastUs = now;

  gx -= m.gBiasX;  gy -= m.gBiasY;

  accelAngles(ax, ay, az, m.accRoll, m.accPitch);
  if (dt > MAX_DT) {
    // 오래 끊긴 뒤에는 자이로 적분이 의미 없으므로 가속도 각도로 다시 시작
    m.roll = m.accRoll;  m.pitch = m.accPitch;
  } else {
    m.roll  = complementary(m.roll,  gx, m.accRoll,  dt);
    m.pitch = complementary(m.pitch, gy, m.accPitch, dt);
  }
  return true;
}

// 시작 각도를 가속도 값으로 맞춤 (0°에서 시작하면 수렴까지 몇 초 걸림)
void initAngles(Imu &m) {
  float ax, ay, az, gx, gy, gz;
  if (readMPU(m.addr, ax, ay, az, gx, gy, gz)) {
    accelAngles(ax, ay, az, m.roll, m.pitch);
  }
  m.lastUs = micros();
}

void printImu(const char *name, Imu &m) {
  Serial.printf("[%s] Roll: %6.1f (acc %6.1f) | Pitch: %6.1f (acc %6.1f)\n",
                name, m.roll, m.accRoll, m.pitch, m.accPitch);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  delay(1000);

  imu1.ok = initMPU(imu1.addr);
  imu2.ok = initMPU(imu2.addr);
  Serial.println(imu1.ok ? "IMU 1 connected! (0x68)" : "IMU 1 connection FAILED!");
  Serial.println(imu2.ok ? "IMU 2 connected! (0x69)" : "IMU 2 connection FAILED!");

  Serial.println("Calibrating... 센서를 움직이지 마세요");
  delay(500);
  if (imu1.ok) imu1.ok = calibrateGyro(imu1);
  if (imu2.ok) imu2.ok = calibrateGyro(imu2);

  initAngles(imu1);
  initAngles(imu2);
  Serial.println("----------------------------");
}

void loop() {
  static uint32_t next = micros() + SAMPLE_US;
  static int count = 0;

  uint32_t now = micros();
  if ((int32_t)(now - next) < 0) return;
  // 출력 등으로 늦어지면 밀린 주기를 몰아서 처리하지 않고 건너뛴다
  next += SAMPLE_US * (1 + (now - next) / SAMPLE_US);

  bool ok1 = imu1.ok && updateImu(imu1, now);
  bool ok2 = imu2.ok && updateImu(imu2, now);

  if (++count < 50) return;   // 0.5초마다 출력
  count = 0;

  if (ok1) printImu("IMU 1", imu1); else Serial.println("[IMU 1] Read failed / not calibrated");
  if (ok2) printImu("IMU 2", imu2); else Serial.println("[IMU 2] Read failed / not calibrated");
  Serial.println("----------------------------");
}
