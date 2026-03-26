package com.example.loader;

import com.example.loader.model.Document;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.*;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.index.CreateIndexParam;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class MilvusService {

    private static final int BATCH_SIZE = 50;

    private final MilvusServiceClient client;
    private final String collectionName;
    private final int dimension;

    public MilvusService(String host, int port, String collectionName, int dimension) {
        System.out.printf("Connecting to Milvus at %s:%d%n", host, port);
        this.client = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost(host)
                        .withPort(port)
                        .build()
        );
        this.collectionName = collectionName;
        this.dimension = dimension;
    }

    public void createCollectionIfNotExists() {
        R<Boolean> exists = client.hasCollection(
                HasCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );
        checkResponse(exists);

        if (Boolean.TRUE.equals(exists.getData())) {
            System.out.println("Collection already exists: " + collectionName);
            return;
        }

        FieldType idField = FieldType.newBuilder()
                .withName("id")
                .withDataType(DataType.Int64)
                .withPrimaryKey(true)
                .withAutoID(false)
                .build();

        FieldType titleField = FieldType.newBuilder()
                .withName("title")
                .withDataType(DataType.VarChar)
                .withMaxLength(500)
                .build();

        FieldType authorField = FieldType.newBuilder()
                .withName("author")
                .withDataType(DataType.VarChar)
                .withMaxLength(200)
                .build();

        FieldType contentField = FieldType.newBuilder()
                .withName("content")
                .withDataType(DataType.VarChar)
                .withMaxLength(5000)
                .build();

        FieldType embeddingField = FieldType.newBuilder()
                .withName("embedding")
                .withDataType(DataType.FloatVector)
                .withDimension(dimension)
                .build();

        R<RpcStatus> createResult = client.createCollection(
                CreateCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withDescription("Document collection for semantic search")
                        .withShardsNum(1)
                        .addFieldType(idField)
                        .addFieldType(titleField)
                        .addFieldType(authorField)
                        .addFieldType(contentField)
                        .addFieldType(embeddingField)
                        .build()
        );
        checkResponse(createResult);
        System.out.println("Collection created: " + collectionName);

        createIndex();
    }

    private void createIndex() {
        R<RpcStatus> indexResult = client.createIndex(
                CreateIndexParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withFieldName("embedding")
                        .withIndexType(IndexType.IVF_FLAT)
                        .withMetricType(MetricType.COSINE)
                        .withExtraParam("{\"nlist\":128}")
                        .build()
        );
        checkResponse(indexResult);
        System.out.println("IVF_FLAT index created on 'embedding' field (COSINE metric)");
    }

    public void insertDocuments(List<Document> documents) {
        if (documents.isEmpty()) {
            System.out.println("No documents to insert.");
            return;
        }

        // Insert in batches
        for (int i = 0; i < documents.size(); i += BATCH_SIZE) {
            List<Document> batch = documents.subList(i, Math.min(i + BATCH_SIZE, documents.size()));
            insertBatch(batch);
            System.out.printf("Inserted batch %d/%d (%d documents)%n",
                    (i / BATCH_SIZE) + 1,
                    (int) Math.ceil((double) documents.size() / BATCH_SIZE),
                    batch.size());
        }

        // Flush to persist data
        R<RpcStatus> flushResult = client.flush(
                io.milvus.param.dml.FlushParam.newBuilder()
                        .addCollectionName(collectionName)
                        .build()
        );
        checkResponse(flushResult);
        System.out.println("Data flushed to disk.");
    }

    private void insertBatch(List<Document> batch) {
        List<Long> ids = batch.stream().map(Document::getId).collect(Collectors.toList());
        List<String> titles = batch.stream().map(Document::getTitle).collect(Collectors.toList());
        List<String> authors = batch.stream().map(Document::getAuthor).collect(Collectors.toList());
        List<String> contents = batch.stream().map(Document::getContent).collect(Collectors.toList());
        List<List<Float>> embeddings = batch.stream().map(Document::getEmbedding).collect(Collectors.toList());

        List<InsertParam.Field> fields = new ArrayList<>();
        fields.add(new InsertParam.Field("id", ids));
        fields.add(new InsertParam.Field("title", titles));
        fields.add(new InsertParam.Field("author", authors));
        fields.add(new InsertParam.Field("content", contents));
        fields.add(new InsertParam.Field("embedding", embeddings));

        R<io.milvus.grpc.MutationResult> insertResult = client.insert(
                InsertParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withFields(fields)
                        .build()
        );
        checkResponse(insertResult);
    }

    public void close() {
        client.close();
    }

    private <T> void checkResponse(R<T> response) {
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus operation failed: " + response.getMessage());
        }
    }
}
