
"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  Check,
  FileText,
  LoaderCircle,
  MessageSquare,
  Plus,
  Send,
  ShieldCheck,
  Sparkles,
  UserRound,
  AlertCircle,
  PanelLeftClose,
} from "lucide-react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const STORAGE_KEY = "knowledgeflow-conversations";


type Source = {
  citation: string;
  document: string;
  page: number | null;
  relevance_score?: number | null;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  citationWarnings?: string[];
};


type Conversation = {
  id: string;
  title: string;
  backendConversationId: string | null;
  messages: ChatMessage[];
  updatedAt: number;
};
type AskResponse = {
  question: string;
  search_question?: string;
  answer: string;
  sources: Source[];
  citation_warnings: string[];
  workflow_status: string;
  conversation_id: string;
  evidence_sufficient?: boolean;
  evidence_reason?: string;
};

function createConversationTitle(question: string): string {
  const cleaned = question.trim().replace(/\s+/g, " ");

  if (!cleaned) {
    return "New conversation";
  }

  return cleaned.length > 40
    ? `${cleaned.slice(0, 40)}...`
    : cleaned;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [hydrated, setHydrated] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  const updateActiveConversation = (
  updates: Partial<Conversation>
) => {
  if (!activeConversationId) return;

  setConversations((previous) =>
    previous.map((conversation) =>
      conversation.id === activeConversationId
        ? {
            ...conversation,
            ...updates,
            updatedAt: Date.now(),
          }
        : conversation
    )
  );
};

useEffect(() => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (!saved) {
      setHydrated(true);
      return;
    }

    const parsed = JSON.parse(saved);

    if (Array.isArray(parsed.conversations)) {
      setConversations(parsed.conversations);
    }

    if (typeof parsed.activeConversationId === "string") {
      setActiveConversationId(parsed.activeConversationId);

      const active = parsed.conversations?.find(
        (conversation: Conversation) =>
          conversation.id === parsed.activeConversationId
      );

      if (active) {
        setMessages(active.messages);
        setConversationId(active.backendConversationId);
      }
    }
  } catch (error) {
    console.error("Failed to restore conversation history:", error);
    localStorage.removeItem(STORAGE_KEY);
  } finally {
    setHydrated(true);
  }
}, []);


  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  
async function handleSubmit(event: FormEvent<HTMLFormElement>) {
  event.preventDefault();

  const text = question.trim();

  if (!text || loading) return;

  setError("");
  setQuestion("");
  setLoading(true);

  const userMessage: ChatMessage = {
    id: crypto.randomUUID(),
    role: "user",
    content: text,
  };

  const existingConversation = conversations.find(
    (item) => item.id === activeConversationId
  );

  const frontendId =
    existingConversation?.id ?? crypto.randomUUID();

  const currentMessages = [
    ...(existingConversation?.messages ?? messages),
    userMessage,
  ];

  const conversation: Conversation = {
    id: frontendId,
    title:
      existingConversation?.title ?? createConversationTitle(text),
    backendConversationId:
      existingConversation?.backendConversationId ?? null,
    messages: currentMessages,
    updatedAt: Date.now(),
  };

  setActiveConversationId(frontendId);
  setMessages(currentMessages);
  setConversations((previous) => [
    conversation,
    ...previous.filter((item) => item.id !== frontendId),
  ]);

  try {
    const response = await fetch(`${API_BASE_URL}/rag/graph/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: text,
        limit: 5,
        ...(conversation.backendConversationId
          ? { conversation_id: conversation.backendConversationId }
          : {}),
      }),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `Backend returned ${response.status}: ${detail.slice(0, 300)}`
      );
    }

    const data: AskResponse = await response.json();

    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: data.answer || "The backend returned an empty answer.",
      sources: data.sources ?? [],
      citationWarnings: data.citation_warnings ?? [],
    };

    const updatedMessages = [...currentMessages, assistantMessage];

    setConversationId(data.conversation_id);
    setMessages(updatedMessages);

    setConversations((previous) => [
      {
        ...conversation,
        backendConversationId: data.conversation_id,
        messages: updatedMessages,
        updatedAt: Date.now(),
      },
      ...previous.filter((item) => item.id !== frontendId),
    ]);
  } catch (err) {
    const message =
      err instanceof Error
        ? err.message
        : "An unexpected error occurred.";

    setError(
      `${message}. Check that FastAPI is running and the API URL is correct.`
    );

    setQuestion(text);
  } finally {
    setLoading(false);
    inputRef.current?.focus();
  }
}


// Auto save
useEffect(() => {
  if (!hydrated) return;

  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversations,
      activeConversationId,
    })
  );
}, [conversations, activeConversationId, hydrated]);

const openConversation = (conversation: Conversation) => {
  if (loading) return;

  setActiveConversationId(conversation.id);
  setMessages(conversation.messages ?? []);
  setConversationId(conversation.backendConversationId ?? null);
  setQuestion("");
  setError("");
};


  const startNewConversation = () => {
  if (loading) return;

  setActiveConversationId(null);
  setMessages([]);
  setConversationId(null);
  setQuestion("");
  setError("");
  inputRef.current?.focus();
};

  return (
    <main className="flex h-dvh overflow-hidden bg-[#f7f8fa] text-slate-900">
      {/* Sidebar */}
      <aside className="hidden w-[270px] shrink-0 flex-col border-r border-slate-200 bg-white p-4 md:flex">
        <div className="mb-8 flex items-center gap-3 px-2 pt-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-sm">
            <Sparkles size={21} />
          </div>
          <div>
            <h1 className="font-semibold tracking-tight">
              KnowledgeFlow AI
            </h1>
            <p className="text-xs text-slate-500">
              Document intelligence
            </p>
          </div>
        </div>

        <button
          onClick={startNewConversation}
          disabled={loading}
          className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus size={18} />
          New conversation
        </button>

        <div className="mt-8">
          <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Workspace
          </p>
        </div>

        
<div className="mt-6 min-h-0 flex-1 overflow-y-auto">
  <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
    Conversation history
  </p>

  {conversations.length === 0 ? (
    <p className="px-2 py-2 text-xs leading-5 text-slate-400">
      Your conversations will appear here.
    </p>
  ) : (
    <div className="space-y-1">
      {[...conversations]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .map((conversation) => (
          <button
            key={conversation.id}
            type="button"
            onClick={() => openConversation(conversation)}
            disabled={loading}
            title={conversation.title}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-sm transition disabled:opacity-50 ${
              activeConversationId === conversation.id
                ? "bg-indigo-50 font-medium text-indigo-700"
                : "text-slate-600 hover:bg-slate-50"
            }`}
          >
            <MessageSquare
              size={16}
              className="shrink-0"
            />

            <span className="min-w-0 flex-1 truncate">
              {conversation.title}
            </span>
          </button>
        ))}
    </div>
  )}
</div>


        <div className="mt-auto rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <ShieldCheck size={17} className="text-emerald-600" />
            Grounded responses
          </div>
          <p className="text-xs leading-5 text-slate-500">
            Answers are generated using retrieved document evidence and
            may include source citations.
          </p>
        </div>
      </aside>

      {/* Main chat panel */}
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[68px] shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600 md:hidden">
              <Sparkles size={19} />
            </div>
            <div>
              <h2 className="text-sm font-semibold">
                Document assistant
              </h2>
              <p className="text-xs text-slate-500">
                Ask questions about your documents
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
  <button
    onClick={startNewConversation}
    disabled={loading}
    className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
  >
    <Plus size={16} />
    <span className="hidden sm:inline">New chat</span>
  </button>

  <div className="hidden items-center gap-2 rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 sm:flex">
    <span className="h-2 w-2 rounded-full bg-emerald-500" />
    RAG assistant
  </div>
</div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center px-5 py-12 sm:px-8">
              <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-lg shadow-indigo-200">
                <Bot size={30} />
              </div>

              <p className="mb-3 text-sm font-medium text-indigo-600">
                YOUR AI DOCUMENT ASSISTANT
              </p>

              <h3 className="max-w-xl text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
                Your documents, understood.
              </h3>

              <p className="mt-4 max-w-xl text-base leading-7 text-slate-500">
                Ask a question about your uploaded documents. KnowledgeFlow
                AI will retrieve relevant passages and use them to build
                a grounded answer.
              </p>

              <div className="mt-9 grid gap-3 sm:grid-cols-2">
                {[
                  {
                    title: "Summarize a policy",
                    description: "What are the key points in this document?",
                    prompt: "Summarize the key points in the employee handbook.",
                  },
                  {
                    title: "Find a specific detail",
                    description: "Look up a rule, date, or requirement.",
                    prompt: "What are the main requirements described in the document?",
                  },
                  {
                    title: "Ask a follow-up",
                    description: "Explore a topic in more detail.",
                    prompt: "Can you explain that in more detail?",
                  },
                  {
                    title: "Check the evidence",
                    description: "Ask what the document actually specifies.",
                    prompt: "What information does the document provide about its main policy?",
                  },
                ].map((item) => (
                  <button
                    key={item.title}
                    onClick={() => {
                      setQuestion(item.prompt);
                      inputRef.current?.focus();
                    }}
                    className="rounded-xl border border-slate-200 bg-white p-4 text-left transition hover:border-indigo-300 hover:bg-indigo-50/40"
                  >
                    <span className="block text-sm font-semibold">
                      {item.title}
                    </span>
                    <span className="mt-1 block text-sm leading-5 text-slate-500">
                      {item.description}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-8 px-4 py-8 sm:px-8">
              {messages.map((message) => (
                <article key={message.id} className="flex gap-3 sm:gap-4">
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                      message.role === "user"
                        ? "bg-slate-200 text-slate-700"
                        : "bg-indigo-600 text-white"
                    }`}
                  >
                    {message.role === "user" ? (
                      <UserRound size={18} />
                    ) : (
                      <Bot size={19} />
                    )}
                  </div>

                  <div className="min-w-0 flex-1 pt-1">
                    <p className="mb-2 text-sm font-semibold">
                      {message.role === "user" ? "You" : "KnowledgeFlow AI"}
                    </p>

                    <div className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-700">
                      {message.content}
                    </div>

                    {message.role === "assistant" &&
                      message.sources &&
                      message.sources.length > 0 && (
                        <div className="mt-5 rounded-xl border border-slate-200 bg-white p-4">
                          <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
                            <FileText size={17} className="text-indigo-600" />
                            Sources
                          </div>

                          <div className="space-y-2">
                            {message.sources.map((source, index) => (
                              <div
                                key={`${source.citation}-${source.document}-${index}`}
                                className="flex items-start gap-3 rounded-lg bg-slate-50 p-3"
                              >
                                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white text-slate-500">
                                  <FileText size={16} />
                                </div>
                                <div className="min-w-0 flex-1">
                                  <p className="break-words text-sm font-medium">
                                    {source.document}
                                  </p>
                                  <p className="mt-1 text-xs text-slate-500">
                                    {source.citation}
                                    {source.page != null
                                      ? ` · Page ${source.page}`
                                      : ""}
                                  </p>
                                </div>
                                <Check
                                  size={16}
                                  className="mt-1 shrink-0 text-emerald-600"
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                    {message.citationWarnings &&
                      message.citationWarnings.length > 0 && (
                        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800">
                          <div className="mb-1 flex items-center gap-2 font-semibold">
                            <AlertCircle size={15} />
                            Answer warnings
                          </div>
                          {message.citationWarnings.map((warning, index) => (
                            <p key={`${warning}-${index}`}>{warning}</p>
                          ))}
                        </div>
                      )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="flex gap-3 sm:gap-4">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white">
                    <Bot size={19} />
                  </div>
                  <div className="pt-2">
                    <p className="mb-2 text-sm font-semibold">
                      KnowledgeFlow AI
                    </p>
                    <div className="flex items-center gap-2 text-sm text-slate-500">
                      <LoaderCircle size={16} className="animate-spin" />
                      Searching documents and preparing an answer...
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Error message */}
        {error && (
          <div className="mx-auto w-full max-w-3xl px-4 sm:px-8">
            <div
              role="alert"
              className="mb-3 flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700"
            >
              <AlertCircle size={18} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Composer */}
        <div className="shrink-0 border-t border-slate-200 bg-white px-4 py-4 sm:px-8 sm:py-5">
          <form
            onSubmit={handleSubmit}
            className="mx-auto max-w-3xl"
          >
            <div className="flex items-end gap-3 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-50">
              <textarea
                ref={inputRef}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="Ask a question about your documents..."
                aria-label="Your question"
                rows={1}
                disabled={loading}
                className="max-h-36 min-h-11 flex-1 resize-y bg-transparent px-3 py-3 text-sm outline-none placeholder:text-slate-400 disabled:opacity-60"
              />

              <button
                type="submit"
                disabled={!question.trim() || loading}
                aria-label="Send message"
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
              >
                {loading ? (
                  <LoaderCircle size={19} className="animate-spin" />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </div>

            <div className="mt-3 flex items-center justify-between gap-3 text-[11px] text-slate-400">
              <span>Enter to send · Shift + Enter for a new line</span>
              <span className="hidden sm:inline">
                Responses are based on retrieved document evidence.
              </span>
            </div>
          </form>
        </div>
      </section>
    </main>
  );
}
