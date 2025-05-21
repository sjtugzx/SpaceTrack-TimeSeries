#include <stdio.h>
#include <math.h>
#include <ctype.h>     // for isdigit in cksum
#include "string.h"
#include <iomanip>
#include <math.h>
#include <malloc.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <fstream>
#include <string>
#include <ctime>
#include <curl.h>
#include <chrono>
#include <thread>
#include <random> // Add random number generator header
#include <regex>

#define bzero(a, b)             memset(a, 0, b)
#pragma warning(disable:4996)
#define MAXPARAM 2048
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "wldap32.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "libcurl_a_debug.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "Normaliz.lib")
#pragma execution_character_set("utf-8")

#define MAX_LINE_LENGTH 100
#define MAX_TLE_COUNT 1500

#define AE 1.0

#ifndef J2000
#define J2000 2451544.5
#endif
#ifndef J1900
#define J1900 (J2000 - 36525.)
#endif

//static int i, isimp;

#define    one (1.)
#define    pi (3.1415926535897932384626)
#define    e6a (1e-6)
#define    pio2 (1.570796326794897)       /* pi/2 */
#define    tothrd (.6666666666666667)    /* 2/3 */
#define    two_thirds   (2. / 3.)
#define    twopi (6.283185307179586)    /* 2pi */
#define    x3pio2 (4.712388980384690)    /* 3pi/2 */
#define    de2ra (.0174532925199433)    /* radians per degree */
#define    ra2de (57.29577951308232)    /* deg per radian */
#define    nocon (4.36332312998582e-3)    /* 2pi / 1440 */
#define    d1con (3.03008550693460e-6)    /* 2pi / 1440^2 */
#define    d2con (2.10422604648236e-9);   /* 2pi / 1440^3 */

#define    ae       1.0

#define    xj2 (1.082616e-3)    /* 2nd gravitational zonal harmonic */
#define    xj3 (-.253881e-5)
#define    xj4 (-1.65597e-6)
#define    ck2 (5.41308E-4)       /* .5 * xj2 */
#define    ck4 (6.2098875E-7)    /* -.375 * xj4 */
#define    xke (.743669161e-1)    /* = (G*M)^(1/2)*(er/min)^(3/2) where G =
            Newton's grav const, M = earth mass */

#define    xkmper (6378.135)    /* equatorial earth radius, km */
            /* SGP4/SGP8 density constants.  qoms2t = ((qo - so) / xkmper) ** 4,
            s = 1 + so / xkmper, where qo = 120 and so = 78 */

#define    qoms2t (1.88027916E-9)
#define    s (1.01222928)

            /* time units/day */
#define xmnpda 1440.0

typedef struct {
    char Satellite_Name[45];
    char Satellite_Number[45];
    char International_Designator[45];
    char Class[2];  // Modified Class length to 2 to include null terminator
    double Epoch;
    double Mean_Anomaly;
    double Right_Ascension_of_Node;
    double Argument_of_Perigee;
    double Eccentricity;
    double Inclination;
    double Mean_Motion;
    double First_Derivative_of_Mean_Motion;
    double Second_Derivative_of_Mean_Motion;
    double BSTAR_Drag_Term;
    int Revolution_Number_at_Epoch;
    int Element_Number;
} TLEData;

int cksum(const char* line)
{
    int tot = 0;      // checksum accumulator
    int count = 68;   // length of data line

    /* accumulate checksum from all but the last char in the line,
    which is the desired checksum */
    do {
        int c;
        c = *line++;
        if (isdigit(c))
            tot += c - '0';
        else if (c == '-')
            tot++;
        // all other chars = 0
    } while (--count);
    return(tot % 10);
}

double sci(const char* string)
{
    char buf[12], * bufptr = buf;

    if (string[1] == ' ')   /* is field blank? */
        return 0.;

    /* get mantissa */
    if (*string == '-')
        *bufptr++ = '-';
    ++string;      /* point to 1st digit of mantissa */
    *bufptr++ = '.';
    strncpy(bufptr, string, 6);   /* mantissa has 5 digits */
    bufptr += 5;
    string += 5;
    *bufptr++ = 'E';

    /* get exponent */
    if (*string == '-')
        *bufptr++ = '-';   /* exponent sign, if neg. */

    *bufptr++ = *++string;      /* copy exponent */
    *bufptr = '\0';
    return(atof(buf));
}

int CheckTLE(const char* str1, const char* str2)
{
    if (cksum(str1) != str1[68]) {
        return -1;
    }
    else if (cksum(str2) != str2[68]) {
        return -2;
    }
    else if (strncmp(str1 + 2, str2 + 2, 5) == 0) {
        if (str1[0] == '1' && str2[0] == '2') {
            return 1;
        }
        else if (str1[0] == '2' && str2[0] == '1') {
            return 2;
        }
        else {
            return 0;
        }
    }
    else {
        return 0;
    }

}

int TLE2Elm(const char* line1, const char* line2, TLEData* tle)
{
    int year, rval;

    if (*line1 != '1' || *line2 != '2') {
        rval = -4;
    }
    else {
        rval = cksum(line1);
    }
    if (rval + '0' != line1[68]) {
        rval = -100;
    }
    else {
        rval = cksum(line2);
    }
    if (rval + '0' != line2[68]) {
        rval = -100;
    }
    else {
        rval = 0;
        char tbuff[80];
        double temp;

        tle->Mean_Anomaly = de2ra * atof(line2 + 43);
        tle->Right_Ascension_of_Node = de2ra * atof(line2 + 17);
        tle->Argument_of_Perigee = de2ra * atof(line2 + 34);
        tle->Inclination = de2ra * atof(line2 + 8);

        memcpy(tbuff, line2 + 25, 11);
        *tbuff = '.';
        tle->Eccentricity = atof(tbuff);

        temp = twopi / (xmnpda * xmnpda);

        memcpy(tbuff, line2 + 52, 11);
        tbuff[11] = '\0';
        tle->Mean_Motion = atof(tbuff) * temp * xmnpda;
        tle->First_Derivative_of_Mean_Motion = atof(line1 + 33) * temp;
        tle->Second_Derivative_of_Mean_Motion = sci(line1 + 44) * temp / xmnpda;

        tle->BSTAR_Drag_Term = sci(line1 + 53) * AE;
        year = (line1[18] - '0') * 10 + (line1[19] - '0');
        if (year < 57)
            year += 100;
        tle->Epoch = atof(line1 + 20) + J1900 + (double)year * 365. + (double)((year - 1) / 4);

        tle->Revolution_Number_at_Epoch = atoi(line2 + 63);
        tle->Element_Number = atoi(line1 + 64);

        strncpy(tle->Satellite_Number, line1 + 2, 5);
        tle->Satellite_Number[5] = '\0';
        tle->Class[0] = line1[7];
        tle->Class[1] = '\0'; // Ensure Class string is properly terminated
        strncpy(tle->International_Designator, line1 + 9, 8);
        tle->International_Designator[8] = '\0';
    }

    return(rval);
}

// Define callback function to handle HTTP response
size_t write_callback(void* ptr, size_t size, size_t nmemb, std::string* data) {
    data->append((char*)ptr, size * nmemb);
    return size * nmemb;
}

// Write to CSV file
void write_csv(const char* filename, const TLEData* tle) {
    std::ofstream file(filename, std::ios::app); // Open file in append mode
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write data
    file << std::fixed << std::setprecision(12);

    file << tle->Satellite_Number << "," << tle->Satellite_Number << "," << tle->International_Designator << ","
        << tle->Class << "," << tle->Epoch << "," // Set output precision
        << tle->Mean_Anomaly << "," << tle->Right_Ascension_of_Node << ","
        << tle->Argument_of_Perigee << "," << tle->Eccentricity << ","
        << tle->Inclination << "," << tle->Mean_Motion << ","
        << tle->First_Derivative_of_Mean_Motion << "," << tle->Second_Derivative_of_Mean_Motion << ","
        << tle->BSTAR_Drag_Term << "," << tle->Revolution_Number_at_Epoch << "," << tle->Element_Number << std::endl;
}

// Function to extract date from URL
std::string extract_date_from_url(const std::string& url) {
    std::regex date_regex(R"(CREATION_DATE/(\d{4}-\d{2}-\d{2})--(\d{4}-\d{2}-\d{2}))");
    std::smatch match;
    if (std::regex_search(url, match, date_regex)) {
        return match[1].str() + "_" + match[2].str();
    }
    return "unknown_date";
}

int main() {
    // Login to get session cookie
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL." << std::endl;
        return 1;
    }

    struct curl_slist* cookies = nullptr;
    std::string login_url = "https://www.space-track.org/ajaxauth/login";
    std::string login_payload = "identity=364739773@qq.com&password=shansixiangssx1722";

    curl_easy_setopt(curl, CURLOPT_URL, login_url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, login_payload.c_str());
    curl_easy_setopt(curl, CURLOPT_COOKIEJAR, "cookies.txt");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, login_payload.size());
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to login: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return 1;
    }

    curl_easy_cleanup(curl);

    std::string save_path_txt = "E:\\TLE_data\\latest_space_data_txt\\";
    std::string save_path_csv = "E:\\TLE_data\\latest_space_data_csv\\";

    // Get current date
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);

    char date_buf[11];
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", now_tm);

    // Create URL with current date
    std::string url = "https://www.space-track.org/basicspacedata/query/class/gp_history/CREATION_DATE/" + 
                      std::string(date_buf) + "--" + std::string(date_buf) + 
                      "/orderby/NORAD_CAT_ID,EPOCH/format/3le/emptyresult/show";

    // Extract date from URL
    std::string date_str = extract_date_from_url(url);

    // Use cookie.txt file for web scraping
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize curl!" << std::endl;
        return 1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    std::string cookie_file = "cookies.txt";
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, cookie_file.c_str());

    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Failed to perform HTTP request: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        return 1;
    }

    curl_easy_cleanup(curl);
    curl_global_cleanup();

    std::string txt_filename = save_path_txt + date_str + "_space_track_data.txt";
    std::ofstream txt_file(txt_filename);
    if (!txt_file.is_open()) {
        std::cerr << "Failed to create text file." << std::endl;
        return 1;
    }
    txt_file << response_data;
    txt_file.close();

    std::string output_csv_path = save_path_csv + date_str + "_space_track_data.csv";

    const char* input_file_path = txt_filename.c_str();
    const char* output_csv_cstr = output_csv_path.c_str();

    FILE* file = fopen(input_file_path, "r");
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    char line1[MAX_LINE_LENGTH];
    char line2[MAX_LINE_LENGTH];
    TLEData tle;

    // Read file line by line
    while (fgets(line1, MAX_LINE_LENGTH, file) && fgets(line2, MAX_LINE_LENGTH, file)) {
        // Parse TLE data
        int parse_result = TLE2Elm(line1, line2, &tle);

        // Check parsing result
        if (parse_result != 0) {
            std::cerr << "TLE parsing failed with error code: " << parse_result << std::endl;
            fclose(file);
            break; // Stop parsing this file
        }

        // Write to CSV file
        write_csv(output_csv_cstr, &tle);
    }

    // Close file
    fclose(file);

    std::cout << "CSV file created successfully!" << std::endl;

    return 0;
}
