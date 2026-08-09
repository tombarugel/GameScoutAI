export interface Env {
  AI: Ai;
}

interface GameConceptRequest {
  cluster_name: string;
  opportunity_score: number;
  average_rating: number;
  game_count: number;
  developer_count: number;
  average_chart_position: number;

  representative_games: string[];
  keywords: string[];

  signal_summary?: string;

  pain_points?: Array<{
    share?: number;
    keywords?: string;
    example_reviews?: string;
  }>;

  concept_style?: "safe" | "differentiated" | "bold";
  previous_titles?: string[];
}


const MODEL = "@cf/google/gemma-4-26b-a4b-it";


// =============================================================================
// CORS + JSON RESPONSE
// =============================================================================

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}


function jsonResponse(
  data: unknown,
  status = 200,
): Response {

  return new Response(
    JSON.stringify(data),
    {
      status,
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders(),
      },
    },
  );
}


// =============================================================================
// PROMPT
// =============================================================================

function buildPrompt(
  data: GameConceptRequest,
): string {

  const styleInstructions = {
    safe:
      "Stay close to validated mechanics in the segment. " +
      "Prioritize feasibility and familiarity.",

    differentiated:
      "Keep validated mechanics but introduce a meaningful twist " +
      "that clearly differentiates the concept.",

    bold:
      "Propose a more original concept that recombines validated " +
      "market signals in a surprising but plausible way.",
  };

  const style =
    data.concept_style ?? "differentiated";

  const painPoints =
    data.pain_points &&
    data.pain_points.length > 0
      ? JSON.stringify(
          data.pain_points,
          null,
          2,
        )
      : "No reliable review pain points available.";
  const previousTitles =
  data.previous_titles &&
  data.previous_titles.length > 0
    ? data.previous_titles.join(", ")
    : "None";
  return `
You are a senior mobile game product strategist.

Your task is to generate ONE original mobile game concept based on
real App Store market evidence.

IMPORTANT RULES:
- Use only the supplied evidence.
- Do not invent revenue, downloads, growth rates or player statistics.
- Do not clone an existing game.
- Do not use protected characters, brands or franchises.
- Representative games are references only.
- If review evidence is missing, do not invent player complaints.
- Keep the concept realistic for a mobile game studio.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not add any text before or after the JSON.

CREATIVE DIVERSITY:
- Treat every request as a fresh ideation attempt.
- Do not default to the most obvious theme suggested by the segment name.
- Explore different settings, fantasies, progression structures and mechanic combinations.
- The title must be distinctive, memorable and specific to the generated concept.
- Avoid generic mobile-game naming patterns.
- Avoid repeatedly using words directly taken from the cluster name or keywords in the title.
- Do not simply combine two market keywords to create the title.
- The title should emerge from the game's fantasy or identity, not from the analytical cluster label.
- Market evidence should constrain the opportunity, not dictate the creative theme.
- Two valid concepts based on the same segment should be able to look and feel substantially different while addressing the same market opportunity.

CONCEPT STYLE:
${styleInstructions[style]}

PREVIOUSLY GENERATED TITLES FOR THIS SEGMENT:
${previousTitles}

DIVERSITY REQUIREMENTS:
- This must be a genuinely new ideation attempt.
- Never reuse a previous title or a close variation of it.
- Avoid distinctive words already used in previous titles.
- Do not simply rearrange words from previous titles.
- Do not build the title directly from the cluster name.
- Avoid generic constructions such as "[Keyword] Merge", "[Keyword] Quest", "[Keyword] Kingdom" unless strongly justified.
- Prefer a title derived from the new game's fantasy, setting or identity.
- The concept itself must also differ meaningfully from previous attempts, not only the title.
- Explore a different setting, fantasy, progression structure or mechanic combination while remaining grounded in the same market evidence.

MARKET SEGMENT:
${data.cluster_name}

OPPORTUNITY SCORE:
${data.opportunity_score}/100

AVERAGE RATING:
${data.average_rating}

NUMBER OF RANKED GAMES:
${data.game_count}

NUMBER OF DEVELOPERS:
${data.developer_count}

AVERAGE CHART POSITION:
${data.average_chart_position}

KEYWORDS:
${data.keywords.join(", ")}

REPRESENTATIVE GAMES:
${data.representative_games.join(", ")}

MARKET SIGNAL:
${data.signal_summary ?? "Not available"}

PLAYER PAIN POINTS:
${painPoints}

Return exactly this JSON structure:

{
  "title": "string",
  "one_line_pitch": "string",
  "genre": "string",
  "target_audience": "string",
  "core_mechanic": "string",
  "core_loop": [
    "step 1",
    "step 2",
    "step 3",
    "step 4"
  ],
  "unique_twist": "string",
  "pain_points_addressed": [
    "string"
  ],
  "retention_ideas": [
    "string",
    "string",
    "string"
  ],
  "monetization": [
    "string",
    "string"
  ],
  "market_rationale": "string",
  "main_risk": "string",
  "originality_score": 0
}

originality_score must be an integer between 0 and 100.
`.trim();
}


// =============================================================================
// JSON EXTRACTION
// =============================================================================

function extractJson(
  text: string,
): unknown {

  let cleaned =
    text.trim();

  // Remove markdown code fences if the model adds them.
  if (
    cleaned.startsWith("```")
  ) {

    cleaned = cleaned
      .replace(
        /^```(?:json)?/i,
        "",
      )
      .replace(
        /```$/i,
        "",
      )
      .trim();
  }

  // Keep only the first complete-looking JSON object.
  const firstBrace =
    cleaned.indexOf("{");

  const lastBrace =
    cleaned.lastIndexOf("}");

  if (
    firstBrace !== -1 &&
    lastBrace !== -1 &&
    lastBrace > firstBrace
  ) {

    cleaned =
      cleaned.slice(
        firstBrace,
        lastBrace + 1,
      );
  }

  return JSON.parse(
    cleaned,
  );
}


// =============================================================================
// AI RESPONSE PARSING
// =============================================================================

function extractGeneratedText(
  aiResponse: unknown,
): string {

  // Direct string response.
  if (
    typeof aiResponse === "string"
  ) {
    return aiResponse;
  }

  if (
    !aiResponse ||
    typeof aiResponse !== "object"
  ) {
    return "";
  }

  const response =
    aiResponse as any;

  // Common Workers AI response.
  if (
    typeof response.response ===
    "string"
  ) {
    return response.response;
  }

  // OpenAI-style chat response.
  if (
    typeof response
      ?.choices?.[0]
      ?.message?.content ===
    "string"
  ) {
    return response
      .choices[0]
      .message
      .content;
  }

  // Sometimes wrapped inside result.
  if (
    typeof response
      ?.result?.response ===
    "string"
  ) {
    return response
      .result
      .response;
  }

  if (
    typeof response
      ?.result?.choices?.[0]
      ?.message?.content ===
    "string"
  ) {
    return response
      .result
      .choices[0]
      .message
      .content;
  }

  // Some models return output_text.
  if (
    typeof response
      ?.output_text ===
    "string"
  ) {
    return response
      .output_text;
  }

  if (
    typeof response
      ?.result?.output_text ===
    "string"
  ) {
    return response
      .result
      .output_text;
  }

  return "";
}


// =============================================================================
// WORKER
// =============================================================================

export default {

  async fetch(
    request: Request,
    env: Env,
  ): Promise<Response> {

    // -------------------------------------------------------------------------
    // CORS preflight
    // -------------------------------------------------------------------------

    if (
      request.method === "OPTIONS"
    ) {

      return new Response(
        null,
        {
          status: 204,
          headers:
            corsHeaders(),
        },
      );
    }


    const url =
      new URL(
        request.url,
      );


    // -------------------------------------------------------------------------
    // Health check
    // -------------------------------------------------------------------------

    if (
      request.method === "GET" &&
      url.pathname === "/health"
    ) {

      return jsonResponse(
        {
          status: "ok",
          service:
            "GameScout AI",
          model:
            MODEL,
        },
      );
    }


    // -------------------------------------------------------------------------
    // Route validation
    // -------------------------------------------------------------------------

    if (
      request.method !== "POST" ||
      url.pathname !== "/generate"
    ) {

      return jsonResponse(
        {
          error:
            "Use POST /generate",
        },
        404,
      );
    }


    // -------------------------------------------------------------------------
    // Parse body
    // -------------------------------------------------------------------------

    let data:
      GameConceptRequest;

    try {

      data =
        await request.json<
          GameConceptRequest
        >();

    } catch {

      return jsonResponse(
        {
          error:
            "Invalid JSON body.",
        },
        400,
      );
    }


    // -------------------------------------------------------------------------
    // Minimal validation
    // -------------------------------------------------------------------------

    if (
      !data.cluster_name ||
      !Array.isArray(
        data.representative_games,
      ) ||
      !Array.isArray(
        data.keywords,
      )
    ) {

      return jsonResponse(
        {
          error:
            "Missing required market data.",
        },
        400,
      );
    }


    // -------------------------------------------------------------------------
    // Build prompt
    // -------------------------------------------------------------------------

    const prompt =
      buildPrompt(
        data,
      );


    // -------------------------------------------------------------------------
    // Workers AI
    // -------------------------------------------------------------------------

    try {

      const aiResponse =
        await env.AI.run(
          MODEL,
          {
            messages: [
              {
                role: "system",
                content:
                  "You are a mobile game product strategist. " +
                  "Return only valid JSON and do not explain your reasoning.",
              },
              {
                role: "user",
                content: prompt,
              },
            ],
            temperature: 0.8,
          },
        );


      const generatedText =
        extractGeneratedText(
          aiResponse,
        );


      // -----------------------------------------------------------------------
      // Empty / unexpected response
      // -----------------------------------------------------------------------

      if (
        !generatedText
      ) {

        console.error(
          "Unable to extract text from AI response:",
          JSON.stringify(
            aiResponse,
          ),
        );

        return jsonResponse(
          {
            error:
              "Unexpected Workers AI response format.",

            debug:
              aiResponse,
          },
          502,
        );
      }


      console.log(
        "GENERATED TEXT:",
        generatedText,
      );


      // -----------------------------------------------------------------------
      // Parse JSON concept
      // -----------------------------------------------------------------------

      let concept:
        unknown;

      try {

        concept =
          extractJson(
            generatedText,
          );

      } catch (
        error
      ) {

        console.error(
          "JSON parsing error:",
          error,
        );

        return jsonResponse(
          {
            error:
              "The AI returned text, but it was not valid JSON.",

            raw_response:
              generatedText,
          },
          502,
        );
      }


      // -----------------------------------------------------------------------
      // Success
      // -----------------------------------------------------------------------

      return jsonResponse(
        {
          success:
            true,

          model:
            MODEL,

          concept:
            concept,
        },
      );

    } catch (
      error
    ) {

      console.error(
        "Workers AI error:",
        error,
      );

      return jsonResponse(
        {
          error:
            "AI generation failed.",

          details:
            String(
              error,
            ),
        },
        500,
      );
    }
  },

} satisfies ExportedHandler<Env>;