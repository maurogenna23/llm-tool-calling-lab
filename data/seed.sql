-- Reference data for Bodegón Aurora. Idempotent: re-running leaves the data unchanged.

INSERT OR IGNORE INTO dishes (name, category, price_cents, description, tags, available) VALUES
    ('Empanadas de osobuco',      'entrada',   890,  'Tres empanadas de osobuco braseado ocho horas, masa criolla.', '', 1),
    ('Provoleta de campo',        'entrada',   950,  'Provoleta a la parrilla con orégano fresco y aceite de oliva.', 'vegetariano,sin-gluten', 1),
    ('Burrata de la casa',        'entrada',  1150,  'Burrata cremosa, tomates asados y albahaca del huerto.', 'vegetariano,sin-gluten', 1),
    ('Rabas con alioli de limón', 'entrada',  1290,  'Calamar rebozado liviano con alioli de limón ahumado.', '', 1),
    ('Bife de chorizo Aurora',    'principal', 2450, 'Bife de 400 g madurado 21 días, papas rústicas y chimichurri.', 'sin-gluten', 1),
    ('Ojo de bife a la sartén',   'principal', 2690, 'Ojo de bife sellado con manteca de hierbas y puré de papa ahumado.', 'sin-gluten', 1),
    ('Risotto de hongos',         'principal', 1890, 'Arroz carnaroli, portobellos, hongos de pino y parmesano.', 'vegetariano,sin-gluten', 1),
    ('Sorrentinos de calabaza',   'principal', 1750, 'Rellenos de calabaza asada y ricota, manteca de salvia.', 'vegetariano', 1),
    ('Trucha patagónica',         'principal', 2290, 'Trucha a la plancha, hinojo encurtido y beurre blanc.', 'sin-gluten', 1),
    ('Curry de garbanzos',        'principal', 1590, 'Garbanzos, leche de coco y arroz jazmín. Va picante.', 'vegano,sin-gluten,picante', 1),
    ('Milanesa napolitana',       'principal', 1690, 'Milanesa de ternera, salsa de tomate, jamón y mozzarella.', '', 0),
    ('Flan mixto',                'postre',     720, 'Flan casero con dulce de leche y crema batida.', 'vegetariano,sin-gluten', 1),
    ('Volcán de chocolate',       'postre',     840, 'Bizcocho tibio de chocolate 70% con helado de crema americana.', 'vegetariano', 1),
    ('Sorbete de maracuyá',       'postre',     620, 'Sorbete cítrico de maracuyá con menta fresca.', 'vegano,sin-gluten', 1),
    ('Malbec Aurora (copa)',      'bebida',     780, 'Malbec de Valle de Uco, cosecha 2023.', 'vegano,sin-gluten', 1),
    ('Limonada de jengibre',      'bebida',     490, 'Limonada de prensado frío con jengibre y romero.', 'vegano,sin-gluten', 1),
    ('Café de especialidad',      'bebida',     420, 'Espresso de origen único, tueste medio.', 'vegano,sin-gluten', 1);

INSERT OR IGNORE INTO dining_tables (id, seats, zone) VALUES
    (1, 2, 'barra'),
    (2, 2, 'barra'),
    (3, 2, 'salon'),
    (4, 2, 'terraza'),
    (5, 4, 'salon'),
    (6, 4, 'salon'),
    (7, 4, 'terraza'),
    (8, 4, 'terraza'),
    (9, 6, 'salon'),
    (10, 6, 'terraza'),
    (11, 8, 'salon'),
    (12, 8, 'salon');

-- 0 = Monday. Closed on Mondays; single service window the rest of the week.
INSERT OR IGNORE INTO opening_hours (weekday, opens_at, closes_at) VALUES
    (0, NULL,    NULL),
    (1, '12:00', '23:30'),
    (2, '12:00', '23:30'),
    (3, '12:00', '23:30'),
    (4, '12:00', '00:30'),
    (5, '12:00', '00:30'),
    (6, '12:00', '23:30');
