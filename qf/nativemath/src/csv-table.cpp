#include "csv-table.h"
#include <fstream>
#include <sstream>

// --- Constructors ---
CSVTable::CSVTable(const std::string& csv_path) {
    std::ifstream file(csv_path);
    std::stringstream buffer;
    if (file) {
        buffer << file.rdbuf();
    }
    file_content_ = std::make_unique<std::string>(buffer.str());
    parse();
}

CSVTable::CSVTable(std::string&& csv_path) : CSVTable(static_cast<const std::string&>(csv_path)) {}

// --- Copy Semantics ---
CSVTable::CSVTable(const CSVTable& other) {
    if (other.file_content_) {
        // Deep copy the underlying string
        file_content_ = std::make_unique<std::string>(*other.file_content_);

        // Helper lambda to calculate the new pointer address for a string_view
        auto rebase_sv = [&](std::string_view sv) {
            size_t offset = sv.data() - other.file_content_->data();
            return std::string_view(file_content_->data() + offset, sv.size());
            };

        // Rebase column names
        column_names_.reserve(other.column_names_.size());
        for (auto sv : other.column_names_) {
            column_names_.push_back(rebase_sv(sv));
        }

        // Rebase all cells
        rows_.reserve(other.rows_.size());
        for (const auto& row : other.rows_) {
            std::vector<std::string_view> new_row;
            new_row.reserve(row.size());
            for (auto sv : row) {
                new_row.push_back(rebase_sv(sv));
            }
            rows_.push_back(std::move(new_row));
        }
    }
}

CSVTable& CSVTable::operator=(const CSVTable& other) {
    if (this != &other) {
        CSVTable temp(other);
        *this = std::move(temp); // Copy-and-swap idiom
    }
    return *this;
}

// --- Move Semantics ---
CSVTable::CSVTable(CSVTable&& other) noexcept
    : file_content_(std::move(other.file_content_)),
    column_names_(std::move(other.column_names_)),
    rows_(std::move(other.rows_)) {
}

CSVTable& CSVTable::operator=(CSVTable&& other) noexcept {
    if (this != &other) {
        file_content_ = std::move(other.file_content_);
        column_names_ = std::move(other.column_names_);
        rows_ = std::move(other.rows_);
    }
    return *this;
}

// --- Parser ---
void CSVTable::parse() {
    if (!file_content_ || file_content_->empty()) return;

    std::string_view content(*file_content_);
    size_t line_start = 0;
    bool is_header = true;

    while (line_start < content.size()) {
        size_t line_end = content.find('\n', line_start);
        std::string_view line;

        if (line_end == std::string_view::npos) {
            line = content.substr(line_start);
            line_start = content.size();
        }
        else {
            line = content.substr(line_start, line_end - line_start);
            line_start = line_end + 1;
        }

        // Strip carriage return if it's a Windows-style CRLF file
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        if (line.empty()) continue;

        std::vector<std::string_view> row;
        size_t cell_start = 0;

        while (cell_start < line.size()) {
            size_t cell_end = line.find(',', cell_start);
            if (cell_end == std::string_view::npos) {
                row.push_back(line.substr(cell_start));
                break;
            }
            else {
                row.push_back(line.substr(cell_start, cell_end - cell_start));
                cell_start = cell_end + 1;
            }
        }

        // Handle trailing comma producing an empty string field
        if (!line.empty() && line.back() == ',') {
            row.push_back(std::string_view(line.data() + line.size(), 0));
        }

        if (is_header) {
            column_names_ = std::move(row);
            is_header = false;
        }
        else {
            rows_.push_back(std::move(row));
        }
    }
}

// --- Accessors & Operations ---
int CSVTable::row_count() {
    return static_cast<int>(rows_.size());
}

int CSVTable::column_count() {
    return static_cast<int>(column_names_.size());
}

std::vector<std::string_view>& CSVTable::column_names() {
    return column_names_;
}

std::optional<std::string_view> CSVTable::operator () (size_t row, size_t column) {
    if (row < rows_.size() && column < rows_[row].size()) {
        return rows_[row][column];
    }
    return std::nullopt;
}

std::optional<std::string_view> CSVTable::operator () (size_t row, const std::string& column_Name) {
    for (size_t c = 0; c < column_names_.size(); ++c) {
        if (column_names_[c] == column_Name) {
            return (*this)(row, c);
        }
    }
    return std::nullopt;
}

std::optional<float> CSVTable::to_float(size_t row, size_t column) {
    auto val = (*this)(row, column);
    if (val) {
        try {
            return std::stod(std::string(*val));
        }
        catch (...) {
            return std::nullopt; // Safe failure on non-float strings
        }
    }
    return std::nullopt;
}

std::optional<float> CSVTable::to_float(size_t row, const std::string& column_Name) {
    auto val = (*this)(row, column_Name);
    if (val) {
        try {
            return std::stod(std::string(*val));
        }
        catch (...) {
            return std::nullopt; // Safe failure on non-float strings
        }
    }
    return std::nullopt;
}

int CSVTable::to_float(size_t row, std::span<const string> columns, std::span<float> destination) {
    // CSVTABLE_FILL_ERR_INVALID_ROW -2: `row` < 0 or row >= this->row_count()
    // Note: row is size_t so it's always >= 0, we just need to check the upper bound.
    if (row >= rows_.size()) {
        return CSVTABLE_FILL_ERR_INVALID_ROW;
    }

    // CSVTABLE_FILL_ERR_INVALIDBUFFER_SIZE -3: `destination.size() < columns.size()`
    if (destination.size() < columns.size()) {
        return CSVTABLE_FILL_ERR_INVALIDBUFFER_SIZE;
    }

    // Verify all columns exist and cache their original indices for efficient lookups
    std::vector<size_t> col_indices;
    col_indices.reserve(columns.size());

    for (const auto& col_name : columns) {
        bool found = false;
        for (size_t c = 0; c < column_names_.size(); ++c) {
            if (column_names_[c] == col_name) {
                found = true;
                col_indices.push_back(c);
                break;
            }
        }

        // CSVTABLE_FILL_ERR_INVALID_COLUMN -1: Some columns in `columns` don't exist 
        if (!found) {
            return CSVTABLE_FILL_ERR_INVALID_COLUMN;
        }
    }

    // CSVTABLE_FILL_OK 0: Fill the destination array with the selected columns
    for (size_t i = 0; i < columns.size(); ++i) {
        std::optional<float> val = to_float(row, col_indices[i]);

        // CSVTABLE_FILL_ERR_NOTFLOAT -4: Cell contains data that cannot be parsed to float
        if (!val.has_value()) {
            return CSVTABLE_FILL_ERR_NOTFLOAT;
        }

        destination[i] = val.value();
    }

    return CSVTABLE_FILL_OK;
}