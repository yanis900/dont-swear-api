import { Hono } from "hono";
import { cors } from "hono/cors";
import { env } from "hono/adapter";
import { Index } from "@upstash/vector";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

type Environment = {
  VECTOR_URL: String;
  VECTOR_TOKEN: String;
};

const app = new Hono();

app.use(cors());

const semanticSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 25,
  separators: [" "],
  chunkOverlap: 8,
});

const PROFANITY_THRESHOLD = 0.86;

app.post("/", async (c) => {
  if (c.req.header("Content-Type") !== "application/json") {
    return c.json({ error: "JSON body expected" }, { status: 406 });
  }

  try {
    const { VECTOR_TOKEN, VECTOR_URL } = env<Environment>(c);

    const index = new Index({
      url: VECTOR_URL,
      token: VECTOR_TOKEN,
      cache: false,
    });

    const body = await c.req.json();
    let { message } = body as { message: string };

    if (!message) {
      return c.json(
        { error: "Message argument is required." },
        { status: 400 }
      );
    }

    if (message.length > 1000) {
      return c.json(
        { error: "Message cannot exceed 1000 characters." },
        { status: 413 }
      );
    }

    const [semanticChunks, wordChunks] = await Promise.all([
      splitTextIntoSemantics(message),
      splitTextIntoWords(message),
    ]);

    const flaggedFor = new Set<{ score: number; text: string }>();

    const vectorRes = await Promise.all([
      ...wordChunks.map(async (wordChunk) => {
        const [vector] = await index.query({
          topK: 1,
          data: wordChunk,
          includeMetadata: true,
        });

        if (vector && vector.score > 0.95) {
          flaggedFor.add({
            text: vector.metadata!.text as string,
            score: vector.score,
          });
        }

        return { score: 0 };
      }),
      ...semanticChunks.map(async (semanticChunk) => {
        const [vector] = await index.query({
          topK: 1,
          data: semanticChunk,
          includeMetadata: true,
        });

        if (vector && vector.score > PROFANITY_THRESHOLD) {
          flaggedFor.add({
            text: vector.metadata!.text as string,
            score: vector.score,
          });
        }

        return vector!;
      }),
    ]);

    if (flaggedFor.size > 0) {
      const sorted = Array.from(flaggedFor).sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0];
      return c.json({
        isProfanity: true,
        score: sorted?.score,
        flaggedFor: sorted?.text,
      });
    } else {
      const mostProfaneChunk = vectorRes.sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0]!;

      return c.json({
        isProfanity: false,
        score: mostProfaneChunk.score,
      });
    }
  } catch (err) {
    console.error(err);

    return c.json(
      { error: "Something went wrong.", err: JSON.stringify(err) },
      { status: 500 }
    );
  }
});

function splitTextIntoWords(text: string): string[] {
  return text.split(/\s/);
}

async function splitTextIntoSemantics(text: string): Promise<string[]> {
  if (text.split(/\s/).length === 1) return [];
  const documents = await semanticSplitter.createDocuments([text]);
  const chunks = documents.map((chunk) => chunk.pageContent);
  return chunks;
}

export default app;
