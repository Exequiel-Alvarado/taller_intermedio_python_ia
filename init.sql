CREATE TABLE ventas (
    id SERIAL PRIMARY KEY,
    producto VARCHAR(100) NOT NULL,
    cantidad INT NOT NULL,
    precio DECIMAL(10, 2) NOT NULL,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO ventas (producto, cantidad, precio) VALUES
('Laptop', 5, 999.99),
('Teclado', 10, 49.99),
('Ratón', 15, 19.99);