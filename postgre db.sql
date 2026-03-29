--
-- PostgreSQL database dump
--

\restrict 0vRE6EOh5RkZmWPCMHuScbt5bnxrcVHVvsVoM4dnfzPPmMIFMCskFM6akLvyA2i

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

-- Started on 2026-03-20 22:12:49

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 221 (class 1259 OID 16632)
-- Name: admins; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.admins (
    admin_id integer NOT NULL,
    firstname character varying(50) NOT NULL,
    lastname character varying(50) NOT NULL,
    password_hash character varying(255) NOT NULL,
    email character varying(100) NOT NULL
);


ALTER TABLE public.admins OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 16631)
-- Name: admins_admin_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.admins_admin_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.admins_admin_id_seq OWNER TO postgres;

--
-- TOC entry 4935 (class 0 OID 0)
-- Dependencies: 220
-- Name: admins_admin_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.admins_admin_id_seq OWNED BY public.admins.admin_id;


--
-- TOC entry 217 (class 1259 OID 16609)
-- Name: documents; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.documents (
    document_id uuid NOT NULL,
    file_name character varying(255) NOT NULL,
    file_path character varying(500) NOT NULL,
    status character varying(20) DEFAULT 'uploaded'::character varying NOT NULL,
    total_chunks integer DEFAULT 0 NOT NULL,
    uploaded_time timestamp with time zone DEFAULT now() NOT NULL,
    indexed_time timestamp with time zone,
    error_message text
);


ALTER TABLE public.documents OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 16620)
-- Name: employees; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.employees (
    emp_id integer NOT NULL,
    firstname character varying(50) NOT NULL,
    lastname character varying(50) NOT NULL,
    email character varying(100) NOT NULL,
    password_hash character varying(255) NOT NULL,
    status character varying(20) DEFAULT 'disabled'::character varying NOT NULL,
    department character varying(50),
    designation character varying(50)
);


ALTER TABLE public.employees OWNER TO postgres;

--
-- TOC entry 218 (class 1259 OID 16619)
-- Name: employees_emp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.employees_emp_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.employees_emp_id_seq OWNER TO postgres;

--
-- TOC entry 4936 (class 0 OID 0)
-- Dependencies: 218
-- Name: employees_emp_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.employees_emp_id_seq OWNED BY public.employees.emp_id;


--
-- TOC entry 223 (class 1259 OID 16643)
-- Name: system_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.system_logs (
    log_id integer NOT NULL,
    actor_type character varying(20) NOT NULL,
    actor_id integer NOT NULL,
    action_type character varying(50) NOT NULL,
    action_description text,
    affected_table character varying(50),
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    ip_address inet,
    status character varying(20)
);


ALTER TABLE public.system_logs OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 16642)
-- Name: system_logs_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.system_logs_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.system_logs_log_id_seq OWNER TO postgres;

--
-- TOC entry 4937 (class 0 OID 0)
-- Dependencies: 222
-- Name: system_logs_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.system_logs_log_id_seq OWNED BY public.system_logs.log_id;


--
-- TOC entry 4761 (class 2604 OID 16635)
-- Name: admins admin_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admins ALTER COLUMN admin_id SET DEFAULT nextval('public.admins_admin_id_seq'::regclass);


--
-- TOC entry 4759 (class 2604 OID 16623)
-- Name: employees emp_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employees ALTER COLUMN emp_id SET DEFAULT nextval('public.employees_emp_id_seq'::regclass);


--
-- TOC entry 4762 (class 2604 OID 16646)
-- Name: system_logs log_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.system_logs ALTER COLUMN log_id SET DEFAULT nextval('public.system_logs_log_id_seq'::regclass);


--
-- TOC entry 4927 (class 0 OID 16632)
-- Dependencies: 221
-- Data for Name: admins; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.admins (admin_id, firstname, lastname, password_hash, email) FROM stdin;
\.


--
-- TOC entry 4923 (class 0 OID 16609)
-- Dependencies: 217
-- Data for Name: documents; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.documents (document_id, file_name, file_path, status, total_chunks, uploaded_time, indexed_time, error_message) FROM stdin;
2367699c-b194-4f87-895d-e3141f512831	doc	fdfghb	uploaded	0	2026-02-18 15:51:39.204376+05:30	\N	\N
10757552-e509-499e-9c6a-94984be35200	STROKX_TECHNOLOGIES_-_Data_Security_Guidelines.pdf	10757552-e509-499e-9c6a-94984be35200\\STROKX_TECHNOLOGIES_-_Data_Security_Guidelines.pdf	indexed	0	2026-03-12 11:32:12.015508+05:30	\N	\N
c77db116-98d7-4dbb-b34b-56e69c95f0b3	STROKX_TECHNOLOGIES_-_Process_Flow_Guidelines.pdf	c77db116-98d7-4dbb-b34b-56e69c95f0b3\\STROKX_TECHNOLOGIES_-_Process_Flow_Guidelines.pdf	indexed	0	2026-03-12 13:49:08.912054+05:30	\N	\N
\.


--
-- TOC entry 4925 (class 0 OID 16620)
-- Dependencies: 219
-- Data for Name: employees; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.employees (emp_id, firstname, lastname, email, password_hash, status, department, designation) FROM stdin;
\.


--
-- TOC entry 4929 (class 0 OID 16643)
-- Dependencies: 223
-- Data for Name: system_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.system_logs (log_id, actor_type, actor_id, action_type, action_description, affected_table, "timestamp", ip_address, status) FROM stdin;
\.


--
-- TOC entry 4938 (class 0 OID 0)
-- Dependencies: 220
-- Name: admins_admin_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.admins_admin_id_seq', 1, false);


--
-- TOC entry 4939 (class 0 OID 0)
-- Dependencies: 218
-- Name: employees_emp_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.employees_emp_id_seq', 1, false);


--
-- TOC entry 4940 (class 0 OID 0)
-- Dependencies: 222
-- Name: system_logs_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.system_logs_log_id_seq', 1, false);


--
-- TOC entry 4771 (class 2606 OID 16641)
-- Name: admins admins_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admins
    ADD CONSTRAINT admins_email_key UNIQUE (email);


--
-- TOC entry 4773 (class 2606 OID 16639)
-- Name: admins admins_firstname_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admins
    ADD CONSTRAINT admins_firstname_key UNIQUE (firstname);


--
-- TOC entry 4775 (class 2606 OID 16637)
-- Name: admins admins_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admins
    ADD CONSTRAINT admins_pkey PRIMARY KEY (admin_id);


--
-- TOC entry 4765 (class 2606 OID 16618)
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (document_id);


--
-- TOC entry 4767 (class 2606 OID 16630)
-- Name: employees employees_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employees
    ADD CONSTRAINT employees_email_key UNIQUE (email);


--
-- TOC entry 4769 (class 2606 OID 16628)
-- Name: employees employees_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employees
    ADD CONSTRAINT employees_pkey PRIMARY KEY (emp_id);


--
-- TOC entry 4777 (class 2606 OID 16651)
-- Name: system_logs system_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.system_logs
    ADD CONSTRAINT system_logs_pkey PRIMARY KEY (log_id);


-- Completed on 2026-03-20 22:12:49

--
-- PostgreSQL database dump complete
--

\unrestrict 0vRE6EOh5RkZmWPCMHuScbt5bnxrcVHVvsVoM4dnfzPPmMIFMCskFM6akLvyA2i

