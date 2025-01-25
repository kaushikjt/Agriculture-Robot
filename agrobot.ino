#define S0 4
#define S1 5
#define S2 6
#define S3 7
#define sensorOut 8

#define BUZZER_HEALTHY 9
#define BUZZER_UNHEALTHY 10

double ch1 = 2;
double ch2 = 3;
int a = 11; // Changed to avoid conflict with S0
int b = 12; // Changed to avoid conflict with S1
int c = 13; // Changed to avoid conflict with S2
int d = 9; // Changed to avoid conflict with S3

void setup() {
  Serial.begin(9600);

  // Initial pin configurations
  pinMode(2, INPUT);
  pinMode(3, INPUT);
  pinMode(a, OUTPUT);
  pinMode(b, OUTPUT);
  pinMode(c, OUTPUT);
  pinMode(d, OUTPUT);

  // Color sensor configurations
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(sensorOut, INPUT);

  // Buzzer configurations
  pinMode(BUZZER_HEALTHY, OUTPUT);
  pinMode(BUZZER_UNHEALTHY, OUTPUT);

  digitalWrite(S0, HIGH); // Frequency scaling to 100%
  digitalWrite(S1, LOW);
}

void loop() {
  // Read pulses from channels
  ch1 = pulseIn(2, HIGH);
  ch2 = pulseIn(3, HIGH);

  if ((ch1 == 0) && (ch2 == 0)) {
    digitalWrite(a, LOW);
    digitalWrite(b, LOW);
    digitalWrite(c, LOW);
    digitalWrite(d, LOW);
  } else if ((ch1 > 1530) && (ch2 > 1530)) {
    digitalWrite(a, HIGH);
    digitalWrite(b, LOW);
    digitalWrite(c, LOW);
    digitalWrite(d, HIGH);
  } else if ((ch1 > 1530) && (ch2 < 1460)) {
    digitalWrite(a, HIGH);
    digitalWrite(b, LOW);
    digitalWrite(c, HIGH);
    digitalWrite(d, LOW);
  } else if ((ch1 < 1460) && (ch2 > 1530)) {
    digitalWrite(a, LOW);
    digitalWrite(b, HIGH);
    digitalWrite(c, LOW);
    digitalWrite(d, HIGH);
  } else if ((ch1 < 1460) && (ch2 < 1460)) {
    digitalWrite(a, LOW);
    digitalWrite(b, HIGH);
    digitalWrite(c, HIGH);
    digitalWrite(d, LOW);
  } else {
    digitalWrite(a, LOW);
    digitalWrite(b, LOW);
    digitalWrite(c, LOW);
    digitalWrite(d, LOW);
  }

  // Read color values from sensor
  int redValue = readColor(LOW, LOW); // Red
  int greenValue = readColor(HIGH, HIGH); // Green
  int blueValue = readColor(LOW, HIGH); // Blue

  Serial.print("R: ");
  Serial.print(redValue);
  Serial.print(" G: ");
  Serial.print(greenValue);
  Serial.print(" B: ");
  Serial.println(blueValue);

  if (isGreenLeaf(redValue, greenValue, blueValue)) {
    Serial.println("Leaf is Healthy (Green)");
    digitalWrite(BUZZER_HEALTHY, HIGH);
    delay(1000);
    digitalWrite(BUZZER_HEALTHY, LOW);
  } else if (isYellowLeaf(redValue, greenValue, blueValue)) {
    Serial.println("Leaf is Unhealthy (Yellow)");
    digitalWrite(BUZZER_UNHEALTHY, HIGH);
    delay(1000);
    digitalWrite(BUZZER_UNHEALTHY, LOW);
  } else {
    Serial.println("Unable to determine leaf health.");
  }

  delay(1000); // Delay for 1 second before next reading
}

int readColor(bool s2, bool s3) {
  digitalWrite(S2, s2);
  digitalWrite(S3, s3);
  delay(100); // Wait for sensor to stabilize

  // Read frequency from sensor output pin
  int frequency = pulseIn(sensorOut, LOW);
  return frequency;
}

bool isGreenLeaf(int red, int green, int blue) {
  // Adjust these values based on your specific sensor and leaf conditions
  bool isGreen = (green < red) && (green < blue);
  if (isGreen) {
    Serial.print("Green Threshold - R: ");
    Serial.print(red);
    Serial.print(" G: ");
    Serial.print(green);
    Serial.print(" B: ");
    Serial.println(blue);
  }
  return isGreen;
}

bool isYellowLeaf(int red, int green, int blue) {
  // Adjust these values based on your specific sensor and leaf conditions
  bool isYellow = (red < green) && (green < blue);
  if (isYellow) {
    Serial.print("Yellow Threshold - R: ");
    Serial.print(red);
    Serial.print(" G: ");
    Serial.print(green);
    Serial.print(" B: ");
    Serial.println(blue);
  }
  return isYellow;
}
