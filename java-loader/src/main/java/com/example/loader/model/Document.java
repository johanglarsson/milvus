package com.example.loader.model;

import java.util.List;

public class Document {
    private long id;
    private String title;
    private String author;
    private String content;
    private List<Float> embedding;

    public Document(long id, String title, String author, String content) {
        this.id = id;
        this.title = title;
        this.author = author;
        this.content = content;
    }

    public long getId() { return id; }
    public String getTitle() { return title; }
    public String getAuthor() { return author; }
    public String getContent() { return content; }
    public List<Float> getEmbedding() { return embedding; }
    public void setEmbedding(List<Float> embedding) { this.embedding = embedding; }

    @Override
    public String toString() {
        return String.format("Document{id=%d, title='%s', author='%s'}", id, title, author);
    }
}
