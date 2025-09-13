import OpenAI from 'openai';
import { NextRequest, NextResponse } from 'next/server';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    let text: string;
    if (file.type === 'application/pdf') {
      const PDFParser = (await import('pdf2json')).default;
      const pdfParser = new PDFParser();
      
      const buffer = await file.arrayBuffer();
      
      const pdfText = await new Promise<string>((resolve, reject) => {
        pdfParser.on('pdfParser_dataError', reject);
        pdfParser.on('pdfParser_dataReady', (pdfData: any) => {
          let extractedText = '';
          pdfData.Pages.forEach((page: any) => {
            page.Texts.forEach((textObj: any) => {
              extractedText += decodeURIComponent(textObj.R[0].T) + ' ';
            });
          });
          resolve(extractedText);
        });
        pdfParser.parseBuffer(Buffer.from(buffer));
      });
      
      text = pdfText;
    } else {
      text = await file.text();
    }
    
    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "Analyze this DME (Durable Medical Equipment) order document and provide a summary in paragraph form only. Output plain text without any formatting, bullet points, lists, or special characters. Focus on equipment requests, approval status, patient needs, and preserve all proper nouns and specific numbers verbatim."
        },
        {
          role: "user",
          content: text
        }
      ],
      max_completion_tokens: 3500,
    });

    return NextResponse.json({ 
      analysis: completion.choices[0].message.content,
      type: 'DME orders'
    });
  } catch (error) {
    return NextResponse.json({ error: 'Analysis failed' }, { status: 500 });
  }
}
