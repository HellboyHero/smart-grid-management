import nodemailer from "nodemailer"

export const transporter = nodemailer.createTransport({
    host: "smtp.gmail.com",
    port: 465,
    secure: true, // true for port 465, false for other ports
    auth: {
      user: "schalla7@gitam.in",
      pass: "qhti nqnr cqwq yqno",
    },
    tls: {
        // do not fail on invalid certs
        rejectUnauthorized: false,
      },
  });