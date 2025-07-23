import express from "express";
import cors from "cors";
import { transporter } from "./transporter.js";

const app = express();

app.use(express.json());
app.use(cors());

app.post('/api/sendmail', async (req, res) => {
    const { email, theftDifference, currentDemandGW, actualDemandMW, predictedGeneration } = req.body;  // Ensure to get email from body directly

    if (!email) {
        return res.status(400).json({ error: "'email' is required" });  // Correct error message
    }

    const mailOptions = {
        from: '"SmartGrid Team 👻" <schalla7@gitam.in>',  // Your email
        to: email,  // Recipient email
        subject: "⚠️ Alert: Potential Power Theft Detected!",
        text: "This is an automated email alert for a potential power theft incident.",
        html: `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>⚠️ Power Theft Alert</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f8f9fa;
                        color: #333;
                        margin: 0;
                        padding: 0;
                    }
                    .container {
                        max-width: 600px;
                        margin: 20px auto;
                        background-color: #fff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    }
                    h1 {
                        color: #d9534f;
                        text-align: center;
                    }
                    p {
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .alert {
                        padding: 10px;
                        background-color: #f9edbe;
                        border: 1px solid #f0e36d;
                        border-radius: 5px;
                        margin: 15px 0;
                    }
                    .alert strong {
                        color: #d9534f;
                    }
                    .footer {
                        margin-top: 30px;
                        text-align: center;
                        font-size: 14px;
                        color: #777;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>⚠️ Power Theft Alert</h1>
                    <p>Dear Team,</p>
                    <p>This is an automated alert regarding a potential power theft incident.</p>
            
                    <div class="alert">
                        <strong>Power Theft Alert:</strong>
                        <ul>
                            <li><strong>Discrepancy Detected:</strong> ${theftDifference} MW</li>
                            <li><strong>Current Demand:</strong> ${currentDemandGW} GW</li>
                            <li><strong>Actual Demand:</strong> ${actualDemandMW} MW</li>
                            <li><strong>Predicted Generation:</strong> ${predictedGeneration} MW</li>
                        </ul>
                    </div>
            
                    <p>Please take immediate action to investigate and address the issue.</p>
            
                    <div class="footer">
                        <p>Best regards,<br>Power Monitoring System</p>
                    </div>
                </div>
            </body>
            </html>
        `,
    };

    try {
        await transporter.sendMail(mailOptions);
        console.log("mail sent")
        return res.status(200).json({
            message: "Email sent successfully",
        });
    } catch (error) {
        console.error("Error sending email:", error);
        return res.status(500).json({
            error: "Failed to send email",
        });
    }
});

const port = 3000;

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
