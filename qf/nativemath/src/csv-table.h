#pragma once
#ifndef __CSV_TABLE__
#define __CSV_TABLE__
#include <string>
#include <vector>
#include <optional>
#include <string_view>
#include <memory>
#include <span>

#define CSVTABLE_FILL_OK    0
#define CSVTABLE_FILL_ERR_INVALID_COLUMN -1
#define CSVTABLE_FILL_ERR_INVALID_ROW -2
#define CSVTABLE_FILL_ERR_INVALIDBUFFER_SIZE -3
#define CSVTABLE_FILL_ERR_NOTFLOAT - 4

using namespace std;

class CSVTable {
private:
    // Storing as a pointer prevents string_view invalidation during move semantics.
    std::unique_ptr<std::string> file_content_;
    std::vector<std::string_view> column_names_;
    std::vector<std::vector<std::string_view>> rows_;
    
    // Helper to parse the CSV file after it is loaded
    void parse();

public:
    CSVTable(const std::string& csv_path);
    CSVTable(std::string&& csv_path);

    // Rule of 5: Explicit Copy and Move semantics
    CSVTable(const CSVTable& other);
    CSVTable& operator=(const CSVTable& other);

    CSVTable(CSVTable&& other) noexcept;
    CSVTable& operator=(CSVTable&& other) noexcept;

    int row_count();
    int column_count();
    std::vector<std::string_view>& column_names();
    std::optional<std::string_view> operator () (size_t row, size_t column);
    std::optional<std::string_view> operator () (size_t row, const std::string& column_Name);

    std::optional<float> to_float(size_t row, size_t column);
    std::optional<float> to_float(size_t row, const std::string& column_Name);


    /* fill_float will fill `destination` buffer with float values
    CSVTABLE_FILL_OK:   0 <= `row` and `row` < `this->row_count()`, all columns in `columns` exist and `destination.size() >= columns.size()`
    CSVTABLE_FILL_ERR_INVALID_COLUMN: -1: Some columns in `columns` don't exist  
    CSVTABLE_FILL_ERR_INVALID_ROW -2: `row` < 0 or row >= this->row_count()
    CSVTABLE_FILL_ERR_INVALIDBUFFER_SIZE -3 `destination.size() < this->column_count()`
    CSVTABLE_FILL_ERR_NOTFLOAT - 4: All columns exist but some of them in the row referenced by `row` is not a floating point number.
    CSV
    */
    int to_float(size_t row, std::span<const string> columns, std::span<float> destination);
};

#endif