#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Konfigurasi WiFi
const char* ssid = "Fauzan";
const char* password = "56473829";

// Konfigurasi Web Server
ESP8266WebServer server(80);

// Konfigurasi Relay
int relayPin = 2;  // GPIO2 (D4 biasanya)

// Konfigurasi LCD (alamat I2C default: 0x27 atau 0x3F)
LiquidCrystal_I2C lcd(0x27, 16, 2);

void handleRoot() {
  server.send(200, "text/plain", "ESP8266 Door Lock Ready");
}

void handleUnlock() {
  Serial.println("📩 Perintah UNLOCK diterima dari client"); // Log masuk
  digitalWrite(relayPin, HIGH);   // Aktifkan relay
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Door Unlocked");
  server.send(200, "text/plain", "Door Unlocked ✅");

  Serial.println("🔓 Pintu terbuka (relay ON)"); // Log ke serial

  delay(5000);                    // Pintu terbuka 5 detik

  digitalWrite(relayPin, LOW);    // Matikan relay
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Door Locked");

  Serial.println("🔒 Pintu terkunci (relay OFF)"); // Log ke serial
}

void setup() {
  Serial.begin(115200);

  // Setup relay
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW);

  // Setup LCD
  Wire.begin(4, 5); // SDA = GPIO4 (D2), SCL = GPIO5 (D1)
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");

  // Koneksi WiFi
  WiFi.begin(ssid, password);
  int dotCount = 0; // untuk animasi titik
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    // Update LCD animasi "Mencari WiFi..."
    lcd.setCursor(0, 1);
    lcd.print("Searching");
    for (int i = 0; i <= dotCount; i++) {
      lcd.print(".");
    }
    dotCount++;
    if (dotCount > 3) dotCount = 0;
  }

  Serial.println("");
  Serial.print("✅ Terhubung ke WiFi. IP Address: ");
  Serial.println(WiFi.localIP());

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("WiFi Connected!");
  lcd.setCursor(0, 1);
  lcd.print(WiFi.localIP().toString()); // tampilkan IP di LCD
  delay(2000);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Door Locked");

  // Setup Web Server
  server.on("/", handleRoot);
  server.on("/unlock", handleUnlock);
  server.begin();
  Serial.println("🌐 HTTP server started");
}

void loop() {
  server.handleClient();
}
