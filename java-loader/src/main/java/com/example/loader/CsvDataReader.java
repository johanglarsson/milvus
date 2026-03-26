package com.example.loader;

import com.example.loader.model.Document;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CsvDataReader {

    public List<Document> read(String filePath) throws IOException, CsvValidationException {
        List<Document> documents = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            String[] headers = reader.readNext(); // skip header row
            if (headers == null) {
                throw new IOException("CSV file is empty: " + filePath);
            }

            String[] line;
            while ((line = reader.readNext()) != null) {
                if (line.length < 4) {
                    System.err.println("Skipping malformed row: " + String.join(",", line));
                    continue;
                }
                long id = Long.parseLong(line[0].trim());
                String title = truncate(line[1].trim(), 500);
                String author = truncate(line[2].trim(), 200);
                String content = truncate(line[3].trim(), 5000);
                documents.add(new Document(id, title, author, content));
            }
        }

        System.out.printf("Read %d documents from %s%n", documents.size(), filePath);
        return documents;
    }

    private String truncate(String value, int maxLength) {
        return value.length() > maxLength ? value.substring(0, maxLength) : value;
    }
}
