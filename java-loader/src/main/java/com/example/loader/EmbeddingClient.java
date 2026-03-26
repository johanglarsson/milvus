package com.example.loader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class EmbeddingClient {

    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final String apiType;
    private final String apiUrl;
    private final String model;
    private final String apiKey;
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;

    public EmbeddingClient(String apiType, String apiUrl, String model, String apiKey) {
        this.apiType = apiType.toLowerCase();
        this.apiUrl = apiUrl;
        this.model = model;
        this.apiKey = apiKey;
        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build();
        this.objectMapper = new ObjectMapper();
    }

    public List<Float> getEmbedding(String text) throws IOException {
        return switch (apiType) {
            case "ollama" -> getOllamaEmbedding(text);
            case "openai" -> getOpenAIEmbedding(text);
            default -> throw new IllegalArgumentException("Unknown embedding API type: " + apiType
                    + ". Supported types: ollama, openai");
        };
    }

    private List<Float> getOllamaEmbedding(String text) throws IOException {
        Map<String, String> body = new HashMap<>();
        body.put("model", model);
        body.put("prompt", text);

        String json = objectMapper.writeValueAsString(body);
        Request request = new Request.Builder()
                .url(apiUrl + "/api/embeddings")
                .post(RequestBody.create(json, JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Ollama API error " + response.code() + ": " + response.body().string());
            }
            JsonNode node = objectMapper.readTree(response.body().string());
            return toFloatList(node.get("embedding"));
        }
    }

    private List<Float> getOpenAIEmbedding(String text) throws IOException {
        Map<String, Object> body = new HashMap<>();
        body.put("model", model);
        body.put("input", text);

        String json = objectMapper.writeValueAsString(body);
        Request request = new Request.Builder()
                .url(apiUrl + "/v1/embeddings")
                .post(RequestBody.create(json, JSON))
                .addHeader("Authorization", "Bearer " + apiKey)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("OpenAI API error " + response.code() + ": " + response.body().string());
            }
            JsonNode node = objectMapper.readTree(response.body().string());
            return toFloatList(node.get("data").get(0).get("embedding"));
        }
    }

    private List<Float> toFloatList(JsonNode arrayNode) {
        if (arrayNode == null || !arrayNode.isArray()) {
            throw new IllegalStateException("Expected an array in embedding response");
        }
        List<Float> result = new ArrayList<>(arrayNode.size());
        for (JsonNode val : arrayNode) {
            result.add(val.floatValue());
        }
        return result;
    }
}
