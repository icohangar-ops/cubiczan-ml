/// RFC-4180 compliant CSV builder.
#[derive(Debug, Clone, Default)]
pub struct CsvBuilder {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl CsvBuilder {
    /// Create a new empty CSV builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder with the given header row.
    pub fn with_headers(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
        }
    }

    /// Add a data row. Panics if the row length doesn't match headers length
    /// (when headers have been set).
    pub fn add_row(&mut self, row: Vec<String>) {
        if !self.headers.is_empty() && row.len() != self.headers.len() {
            panic!(
                "Row length {} does not match header length {}",
                row.len(),
                self.headers.len()
            );
        }
        self.rows.push(row);
    }

    /// Build the CSV string with proper RFC-4180 escaping.
    ///
    /// - Fields containing commas, double-quotes, or newlines are wrapped in double-quotes
    /// - Internal double-quotes are doubled (escaped as "")
    pub fn to_string(&self) -> String {
        let mut output = String::new();

        if !self.headers.is_empty() {
            let header_line: Vec<String> = self.headers.iter().map(|f| escape_field(f)).collect();
            output.push_str(&header_line.join(","));
            output.push('\n');
        }

        for row in &self.rows {
            let line: Vec<String> = row.iter().map(|f| escape_field(f)).collect();
            output.push_str(&line.join(","));
            output.push('\n');
        }

        output
    }

    /// Get the number of data rows (excluding header).
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Check if the builder has any data (rows or headers).
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty() && self.rows.is_empty()
    }

    /// Get the number of columns (from headers or first row).
    pub fn column_count(&self) -> usize {
        if !self.headers.is_empty() {
            self.headers.len()
        } else if !self.rows.is_empty() {
            self.rows[0].len()
        } else {
            0
        }
    }

    /// Add a row without validating column count.
    pub fn add_row_unchecked(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }
}

/// Escape a single CSV field per RFC-4180.
fn escape_field(field: &str) -> String {
    let needs_quoting = field.contains(',')
        || field.contains('"')
        || field.contains('\n')
        || field.contains('\r');

    if needs_quoting {
        let escaped = field.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        field.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let builder = CsvBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.row_count(), 0);
    }

    #[test]
    fn test_with_headers() {
        let builder = CsvBuilder::with_headers(vec!["a".to_string(), "b".to_string()]);
        assert!(!builder.is_empty());
        assert_eq!(builder.column_count(), 2);
        assert_eq!(builder.row_count(), 0);
    }

    #[test]
    fn test_add_row() {
        let mut builder = CsvBuilder::with_headers(vec!["name".to_string(), "age".to_string()]);
        builder.add_row(vec!["Alice".to_string(), "30".to_string()]);
        assert_eq!(builder.row_count(), 1);
    }

    #[test]
    fn test_simple_csv_output() {
        let mut builder = CsvBuilder::with_headers(vec!["name".to_string(), "age".to_string()]);
        builder.add_row(vec!["Alice".to_string(), "30".to_string()]);
        builder.add_row(vec!["Bob".to_string(), "25".to_string()]);
        let csv = builder.to_string();
        assert_eq!(csv, "name,age\nAlice,30\nBob,25\n");
    }

    #[test]
    fn test_escape_comma() {
        let mut builder = CsvBuilder::with_headers(vec!["desc".to_string()]);
        builder.add_row(vec!["Hello, World".to_string()]);
        let csv = builder.to_string();
        assert_eq!(csv, "desc\n\"Hello, World\"\n");
    }

    #[test]
    fn test_escape_quotes() {
        let mut builder = CsvBuilder::with_headers(vec!["quote".to_string()]);
        builder.add_row(vec!["say \"hi\"".to_string()]);
        let csv = builder.to_string();
        assert!(csv.contains("\"say \"\"hi\"\"\""));
    }

    #[test]
    fn test_escape_newline() {
        let mut builder = CsvBuilder::with_headers(vec!["text".to_string()]);
        builder.add_row(vec!["line1\nline2".to_string()]);
        let csv = builder.to_string();
        assert!(csv.starts_with("text\n\"line1\nline2\"\n"));
    }

    #[test]
    fn test_no_headers() {
        let mut builder = CsvBuilder::new();
        builder.add_row_unchecked(vec!["1".to_string(), "2".to_string()]);
        let csv = builder.to_string();
        assert_eq!(csv, "1,2\n");
    }

    #[test]
    fn test_empty_rows() {
        let builder = CsvBuilder::with_headers(vec!["a".to_string()]);
        let csv = builder.to_string();
        assert_eq!(csv, "a\n");
    }

    #[test]
    #[should_panic(expected = "Row length")]
    fn test_mismatched_row_length() {
        let mut builder = CsvBuilder::with_headers(vec!["a".to_string(), "b".to_string()]);
        builder.add_row(vec!["only_one".to_string()]);
    }

    #[test]
    fn test_column_count_no_headers() {
        let mut builder = CsvBuilder::new();
        assert_eq!(builder.column_count(), 0);
        builder.add_row_unchecked(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
        assert_eq!(builder.column_count(), 3);
    }

    #[test]
    fn test_roundtrip_escape() {
        let mut builder = CsvBuilder::with_headers(vec!["field".to_string()]);
        builder.add_row(vec!["a\"b,c\nd".to_string()]);
        let csv = builder.to_string();
        // Should be parseable: header line + data line with escaped field
        assert!(csv.contains("\"\"")); // doubled quotes
        assert!(csv.starts_with("field\n"));
    }

    #[test]
    fn test_large_dataset() {
        let mut builder = CsvBuilder::with_headers(vec!["id".to_string(), "value".to_string()]);
        for i in 0..1000 {
            builder.add_row(vec![i.to_string(), format!("val_{}", i)]);
        }
        assert_eq!(builder.row_count(), 1000);
        let csv = builder.to_string();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 1001); // 1 header + 1000 data
    }
}
