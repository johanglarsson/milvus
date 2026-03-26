package com.example.loader;

import com.example.loader.model.Document;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

public class MilvusDataLoader {

    public static void main(String[] args) throws Exception {
        Properties config = loadConfig(args);

        String milvusHost = config.getProperty("milvus.host", "localhost");
        int milvusPort = Integer.parseInt(config.getProperty("milvus.port", "19530"));
        String collectionName = config.getProperty("milvus.collection", "documents");

        String embeddingApiType = config.getProperty("embedding.api.type", "ollama");
        String embeddingApiUrl = config.getProperty("embedding.api.url", "http://localhost:11434");
        String embeddingModel = config.getProperty("embedding.model", "nomic-embed-text");
        String embeddingApiKey = config.getProperty("embedding.api.key", "");
        int embeddingDimension = Integer.parseInt(config.getProperty("embedding.dimension", "768"));

        String dataFilePath = config.getProperty("data.file.path", "../data/sample_documents.csv");

        System.out.println("=== Milvus Data Loader ===");
        System.out.println("Collection : " + collectionName);
        System.out.println("Embedding  : " + embeddingApiType + " / " + embeddingModel + " (dim=" + embeddingDimension + ")");
        System.out.println("Data file  : " + dataFilePath);
        System.out.println();

        // 1. Read CSV
        CsvDataReader reader = new CsvDataReader();
        List<Document> documents = reader.read(dataFilePath);

        // 2. Generate embeddings
        EmbeddingClient embeddingClient = new EmbeddingClient(embeddingApiType, embeddingApiUrl, embeddingModel, embeddingApiKey);
        System.out.println("Generating embeddings for " + documents.size() + " documents...");
        for (int i = 0; i < documents.size(); i++) {
            Document doc = documents.get(i);
            // Embed title + content together for richer representation
            String textToEmbed = doc.getTitle() + ". " + doc.getContent();
            List<Float> embedding = embeddingClient.getEmbedding(textToEmbed);
            doc.setEmbedding(embedding);
            System.out.printf("  [%d/%d] Embedded: %s%n", i + 1, documents.size(), doc.getTitle());
        }

        // 3. Insert into Milvus
        MilvusService milvusService = new MilvusService(milvusHost, milvusPort, collectionName, embeddingDimension);
        try {
            milvusService.createCollectionIfNotExists();
            milvusService.insertDocuments(documents);
            System.out.println();
            System.out.println("Done! " + documents.size() + " documents loaded into collection '" + collectionName + "'.");
        } finally {
            milvusService.close();
        }
    }

    private static Properties loadConfig(String[] args) throws IOException {
        Properties props = new Properties();

        // First load defaults from classpath
        try (InputStream defaultStream = MilvusDataLoader.class.getClassLoader()
                .getResourceAsStream("application.properties")) {
            if (defaultStream != null) {
                props.load(defaultStream);
            }
        }

        // Allow overriding via external file passed as first argument
        if (args.length > 0) {
            System.out.println("Loading config from: " + args[0]);
            try (FileInputStream externalStream = new FileInputStream(args[0])) {
                props.load(externalStream);
            }
        }

        // Allow overriding via environment variables (e.g. MILVUS_HOST)
        overrideFromEnv(props, "milvus.host", "MILVUS_HOST");
        overrideFromEnv(props, "milvus.port", "MILVUS_PORT");
        overrideFromEnv(props, "milvus.collection", "MILVUS_COLLECTION");
        overrideFromEnv(props, "embedding.api.type", "EMBEDDING_API_TYPE");
        overrideFromEnv(props, "embedding.api.url", "EMBEDDING_API_URL");
        overrideFromEnv(props, "embedding.model", "EMBEDDING_MODEL");
        overrideFromEnv(props, "embedding.api.key", "EMBEDDING_API_KEY");
        overrideFromEnv(props, "embedding.dimension", "EMBEDDING_DIMENSION");
        overrideFromEnv(props, "data.file.path", "DATA_FILE_PATH");

        return props;
    }

    private static void overrideFromEnv(Properties props, String key, String envVar) {
        String value = System.getenv(envVar);
        if (value != null && !value.isBlank()) {
            props.setProperty(key, value);
        }
    }
}
